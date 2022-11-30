
    # def loss_labels(self, outputs, targets, indices, _, num_boxes, log=True):
    #     """
    #     Classification loss (NLL)
        
    #     Arguments:
    #     ----------
    #     outputs : dict
    #         - "pred_class_logits" : Tensor [bs, q , 2]
    #     targets : list[dict]
    #         - "labels" : Tensor [bs, q]
    #     indices : list of tuples -- len(indices) = bs
    #         - [(out_idx, tgt_idx), ...]
    #     """
        
    #     assert "pred_class_logits" in outputs
        
    #     outputs_logits = outputs["pred_class_logits"] # [bs, q, 2]
    #     bs, q, _ = outputs_logits.shape

    #     idx = self._get_src_permutation_idx(indices) # [q*bs, b_idx], [q*bs, src_idx]
    #     target_classes_o = torch.cat([t["labels"][tgt_idx] for t, (_, tgt_idx) in zip(targets, indices)]) # [bs*q]
    #     target_classes = torch.full(outputs_logits.shape[:2], 0,
    #                                 dtype=torch.int64, device=outputs_logits.device) # [bs, q] Where all classes point to the no-object class
    #     target_classes[idx] = target_classes_o # [bs, q]
        

    #     loss_ce = focal_loss(outputs_logits, target_classes, alpha=self.focal_alpha, gamma=2.0, reduction="none") # [bs, q]
    #     loss_ce = loss_ce.view(bs, q, -1) # [bs, q, 2]
    #     loss_ce = (loss_ce.mean(1).sum() / num_boxes) * outputs_logits.shape[1]
        

    #     losses = {"loss_ce": loss_ce}
    #     stats = {}
    #     if log:
    #         stats = {"loss_ce": loss_ce.detach()}
    #         predicted_bg = (outputs_logits.argmax(-1) == 0).sum()
    #         predicted_obj = (outputs_logits.argmax(-1) == 1).sum()

    #         prec, acc, rec = prec_acc_rec(outputs_logits.softmax(dim=-1), target_classes)
    #         stats = {"class_acc": acc, "class_prec": prec, "class_rec":rec}
    #         #stats.update({"predicted_bg": predicted_bg, "predicted_obj": predicted_obj})
    #         stats.update(losses)
    #     return losses, stats
    
    
    
    # def loss_similarity(self, outputs, targets, indices, _, num_boxes, log=True):
    #     """
    #     Similarity loss (NLL)
        
    #     Arguments:
    #     ----------
    #     outputs : dict
    #         - "pred_sim_logits" : Tensor [bs, q , 2]
    #     targets : list[dict]
    #         - "labels" : Tensor [bs, q]
    #     indices : list of tuples -- len(indices) = bs
    #         - [(out_idx, tgt_idx), ...]
    #     """
        
    #     assert "pred_sim_logits" in outputs
        
    #     outputs_logits = outputs["pred_sim_logits"]
        
        
    #     bs, q, _ = outputs_logits.shape
    #     idx = self._get_src_permutation_idx(indices) # [q*bs, b_idx], [q*bs, src_idx]
    #     target_classes_o = torch.cat([t["sim_labels"][tgt_idx] for t, (_, tgt_idx) in zip(targets, indices)]) # [bs*q]
    #     target_classes = torch.full(outputs_logits.shape[:2], 0,
    #                                 dtype=torch.int64, device=outputs_logits.device) # [bs, q] Where all classes point to the no-object class
    #     target_classes[idx] = target_classes_o # [bs, q]
        
    #     loss_sim = focal_loss(outputs_logits, target_classes, alpha=self.focal_alpha, gamma=2.0, reduction="none") # [bs, q]
    #     loss_sim = loss_sim.view(bs, q, -1) # [bs, q, 2]
    #     loss_sim = (loss_sim.mean(1).sum() / num_boxes) * outputs_logits.shape[1]
        
    #     losses = {"loss_sim": loss_sim}
    #     stats = {}
    #     if log:
    #         stats = {"loss_sim": loss_sim.detach()}
    #         predicted_bg = (outputs_logits.argmax(-1) == 0).sum()
    #         predicted_obj = (outputs_logits.argmax(-1) == 1).sum()

    #         prec, acc, rec = prec_acc_rec(outputs_logits.softmax(dim=-1), target_classes)
    #         stats = {"similarity_acc": acc, "similarity_prec": prec, "similarity_rec":rec}
    #         #stats.update({"sim_bg": predicted_bg, "sim_obj": predicted_obj})
    #         stats.update(losses)
    #     return losses, stats
    
    
    
    
    
    
    
    
    
    
    
    
     def loss_similarity(self, outputs, targets, indices, indices_2ndbest, num_boxes, log=True):
        """
        Similarity loss (NLL)
        
        Arguments:
        ----------
        outputs : dict
            - "pred_sim_logits" : Tensor [bs, q , 2]
        targets : list[dict]
            - "labels" : Tensor [bs, q]
        indices : list of tuples -- len(indices) = bs
            - [(out_idx, tgt_idx), ...]
        """
        
        assert "pred_sim_logits" in outputs
        outputs_logits = outputs["pred_sim_logits"]
        
        base_loss = self.base_loss
        if base_loss:
            prefix = "base_"
        else:
            prefix = ""

        bs, q, _ = outputs_logits.shape
        idx = self._get_src_permutation_idx(indices) # [q*bs, b_idx], [q*bs, src_idx]
        
        target_classes_o = torch.cat([t[f"{prefix}sim_labels"][tgt_idx] for t, (_, tgt_idx) in zip(targets, indices)]) # [bs*q]
        target_classes = torch.full(outputs_logits.shape[:2], 0,
                                    dtype=torch.int64, device=outputs_logits.device) # [bs, q] Where all classes point to the no-object class
        target_classes[idx] = target_classes_o # [bs, q]
        
        out_ = outputs_logits
        tgt_ = target_classes
        
        if base_loss:
            b_idx = []
            src_idx = []
            for ind in indices_2ndbest:
                b_, src_ = self._get_src_permutation_idx(ind)
                b_idx.append(b_)
                src_idx.append(src_)
            
            idx_2nd = torch.cat(src_idx, dim=0)
            b_2nd = torch.cat(b_idx, dim=0)
            idx_2 = (b_2nd, idx_2nd)
        
            target_classes_2nd = torch.zeros(len(idx_2nd), dtype=torch.int64, device=outputs_logits.device)
           
            out_ = torch.cat([outputs_logits[idx], outputs_logits[idx_2]], dim=0)
            tgt_ = torch.cat([target_classes_o, target_classes_2nd], dim=0)
        
        loss_sim = focal_loss(out_, tgt_, alpha=self.focal_alpha, gamma=2.0, reduction="none") # [bs, q]
        
        if not base_loss:
            loss_sim = loss_sim.view(bs, q, -1) # [bs, q, 2]
            loss_sim = (loss_sim.mean(1).sum() / num_boxes) * outputs_logits.shape[1]
        else:
            loss_sim = 10*loss_sim.sum() / tgt_.sum()
        
        losses = {"loss_sim": loss_sim}
        stats = {}
        if log:
            stats = {"loss_sim": loss_sim.detach()}
            prec, acc, rec = prec_acc_rec(outputs_logits.softmax(dim=-1), target_classes)
            stats = {"similarity_acc": acc, "similarity_prec": prec, "similarity_rec":rec}
            stats.update(losses)
        return losses, stats
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        def loss_labels(self, outputs, targets, indices, indices_2ndbest, num_boxes, log=True):
        """
        Classification loss (NLL)
        
        Arguments:
        ----------
        outputs : dict
            - "pred_class_logits" : Tensor [bs, q , 2]
        targets : list[dict]
            - "labels" : Tensor [bs, q]
        indices : list of tuples -- len(indices) = bs
            - [(out_idx, tgt_idx), ...]
        """
        
        assert "pred_class_logits" in outputs
        outputs_logits = outputs["pred_class_logits"] # [bs, q, 2]
        
        base_loss = self.base_loss
        if base_loss:
            prefix = "base_"
        else:
            prefix = ""

        bs, q, _ = outputs_logits.shape
        idx = self._get_src_permutation_idx(indices) # [q*bs, b_idx], [q*bs, src_idx]

        target_classes_o = torch.cat([t[f"{prefix}labels"][tgt_idx] for t, (_, tgt_idx) in zip(targets, indices)]) # [bs*q]
        target_classes = torch.full(outputs_logits.shape[:2], 0,
                                    dtype=torch.int64, device=outputs_logits.device) # [bs, q] Where all classes point to the no-object class
        target_classes[idx] = target_classes_o # [bs, q]
        
        out_ = outputs_logits
        tgt_ = target_classes
        
        out_tgt = [(out_, tgt_)]
        
        if base_loss:
            b_idx = []
            src_idx = []
            for ind in indices_2ndbest:
                idx_ = self._get_src_permutation_idx(ind)
                out_ = outputs_logits[idx_]
                tgt_ = torch.zeros(len(idx_[0]), dtype=torch.int64, device=outputs_logits.device)
                out_tgt.append((out_, tgt_))
            
        #     idx_2nd = torch.cat(src_idx, dim=0)
        #     b_2nd = torch.cat(b_idx, dim=0)
        #     idx_2 = (b_2nd, idx_2nd)
        
        #     target_classes_2nd = torch.zeros(len(idx_2nd), dtype=torch.int64, device=outputs_logits.device)
           
        #     out_ = torch.cat([outputs_logits[idx], outputs_logits[idx_2]], dim=0)
        #     tgt_ = torch.cat([target_classes_o, target_classes_2nd], dim=0)
        
        # loss_ce = focal_loss(out_, tgt_, alpha=self.focal_alpha, gamma=2.0, reduction="none") # [bs, q]
        
        # if not base_loss:
        #     loss_ce = loss_ce.view(bs, q, -1) # [bs, q, 2]
        #     loss_ce = (loss_ce.mean(1).sum() / num_boxes) * outputs_logits.shape[1]
        # else:
        #     loss_ce = 10*loss_ce.sum() / tgt_.sum()
        
        if not base_loss:
            loss_ce = focal_loss(out_, tgt_, alpha=self.focal_alpha, gamma=2.0, reduction="none") # [bs, q]
            loss_ce = loss_ce.view(bs, q, -1) # [bs, q, 2]
            loss_ce = (loss_ce.mean(1).sum() / num_boxes) * outputs_logits.shape[1]
        else:
            loss_batch = []
            coef = out_tgt[0][1].sum()
            for i, (out_, tgt_) in enumerate(out_tgt):
                loss_ce = focal_loss(out_, tgt_, alpha=self.focal_alpha, gamma=2.0, reduction="none")
                #coef = tgt_.sum() if i==0 else torch.ones_like(tgt_).sum()
                loss_batch.append(loss_ce.sum() / coef)
                
            loss_ce = torch.stack(loss_batch).mean()
        
        losses = {"loss_ce": loss_ce}
        stats = {}
        if log:
            stats = {"loss_ce": loss_ce.detach()}
            prec, acc, rec = prec_acc_rec(outputs_logits.softmax(dim=-1), target_classes)
            stats = {"class_acc": acc, "class_prec": prec, "class_rec":rec}
            stats.update(losses)
        return losses, stats