class RNA_Model(pl.LightningModule):

    def __init__(self, args):
        super(TriviaQA, self).__init__()

        self.model = self.load_model()
        self.num_labels = 2
        self.config=LongformerConfig.from_pretrained('longformer-base-4096/') 
        self.qa_outputs = torch.nn.Linear(self.model.config.hidden_size, self.num_labels)
        

    def load_model(self):
        model = Longformer.from_pretrained(args.model_path)
        for layer in model.encoder.layer:
            layer.attention.self.attention_mode = self.args.attention_mode
            self.args.attention_window = layer.attention.self.attention_window

        print("Loaded model with config:")
        print(model.config)

        for p in model.parameters():
            p.requires_grad_(True)
        model.train()
        return model

    def forward(self, input_ids, attention_mask, segment_ids, start_positions, end_positions):
        question_end_index = self._get_question_end_index(input_ids)
        # Each batch is one document, and each row of the batch is a chunck of the document.
        # Make sure all rows have the same question length.
        assert (question_end_index[0].float() == question_end_index.float().mean()).item()

        # local attention everywhere
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        # global attention for the question tokens
        attention_mask[:, :question_end_index.item()] = 2

        # sliding_chunks implemenation of selfattention requires that seqlen is multiple of window size
        input_ids, attention_mask = pad_to_window_size(
            input_ids, attention_mask, self.args.attention_window, self.tokenizer.pad_token_id)

        outputs = self.model(
                input_ids,
                attention_mask=attention_mask)[0]

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)

    def or_softmax_cross_entropy_loss_one_doc(self, logits, target, ignore_index=-1, dim=-1):
        """loss function suggested in section 2.2 here https://arxiv.org/pdf/1710.10723.pdf"""
        assert logits.ndim == 2
        assert target.ndim == 2
        assert logits.size(0) == target.size(0)

        # with regular CrossEntropyLoss, the numerator is only one of the logits specified by the target
        # here, the numerator is the sum of a few potential targets, where some of them is the correct answer

        # compute a target mask
        target_mask = target == ignore_index
        # replaces ignore_index with 0, so `gather` will select logit at index 0 for the msked targets
        masked_target = target * (1 - target_mask.long())
        # gather logits
        gathered_logits = logits.gather(dim=dim, index=masked_target)
        # Apply the mask to gathered_logits. Use a mask of -inf because exp(-inf) = 0
        gathered_logits[target_mask] = float('-inf')

        # each batch is one example
        gathered_logits = gathered_logits.view(1, -1)
        logits = logits.view(1, -1)

        # numerator = log(sum(exp(gathered logits)))
        log_score = torch.logsumexp(gathered_logits, dim=dim, keepdim=False)
        # denominator = log(sum(exp(logits)))
        log_norm = torch.logsumexp(logits, dim=dim, keepdim=False)

        # compute the loss
        loss = -(log_score - log_norm)

        # some of the examples might have a loss of `inf` when `target` is all `ignore_index`.
        # remove those from the loss before computing the sum. Use sum instead of mean because
        # it is easier to compute
        return loss[~torch.isinf(loss)].sum()

    def training_step(self, batch, batch_nb):
        input_ids, input_mask, segment_ids, subword_starts, subword_ends, qids, aliases = batch
        output = self.forward(input_ids, input_mask, segment_ids, subword_starts, subword_ends)
        loss = output[0]
        lr = loss.new_zeros(1) + self.trainer.optimizers[0].param_groups[0]['lr']
        tensorboard_logs = {'train_loss': loss, 'lr': lr,
                            'input_size': input_ids.numel(),
                            'mem': torch.cuda.memory_allocated(input_ids.device) / 1024 ** 3}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        input_ids, input_mask, segment_ids, subword_starts, subword_ends, qids, aliases = batch
        output = self.forward(input_ids, input_mask, segment_ids, subword_starts, subword_ends)
        loss, start_logits, end_logits = output[:3]
        answers = self.decode(input_ids, start_logits, end_logits)

        # each batch is one document
        answers = sorted(answers, key=lambda x: x['score'], reverse=True)[0:1]
        qids = [qids]
        aliases = [aliases]

        f1_scores = [evaluation_utils.metric_max_over_ground_truths(evaluation_utils.f1_score, answer['text'],
                                                                    aliase_list)
                     for answer, aliase_list in zip(answers, aliases)]
        # TODO: if slow, skip em_scores, and use (f1_score == 1.0) instead
        em_scores = [evaluation_utils.metric_max_over_ground_truths(evaluation_utils.exact_match_score, answer['text'],
                                                                    aliase_list)
                     for answer, aliase_list in zip(answers, aliases)]
        answer_scores = [answer['score'] for answer in answers]  # start_logit + end_logit
        assert len(answer_scores) == len(f1_scores) == len(em_scores) == len(qids) == len(aliases) == 1

        # TODO: delete this metric
        pred_subword_starts = start_logits.argmax(dim=1)
        pred_subword_ends = end_logits.argmax(dim=1)
        exact_match = (subword_ends[:, 0].squeeze(dim=-1) == pred_subword_ends).float() *  \
                      (subword_starts[:, 0].squeeze(dim=-1) == pred_subword_starts).float()

        return {'vloss': loss, 'vem': exact_match.mean(),
                'qids': qids, 'answer_scores': answer_scores,
                'f1': f1_scores, 'em': em_scores}

    

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None):
        optimizer.step()
        optimizer.zero_grad()
        self.scheduler.step(self.global_step)

    def configure_optimizers(self):
        def lr_lambda(current_step):
            if current_step < self.args.warmup:
                return float(current_step) / float(max(1, self.args.warmup))
            return max(0.0, float(self.args.steps - current_step) / float(max(1, self.args.steps - self.args.warmup)))
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=-1)  # scheduler is not saved in the checkpoint, but global_step is, which is enough to restart
        self.scheduler.step(self.global_step)

