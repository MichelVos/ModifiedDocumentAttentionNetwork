#  Copyright Université de Rouen Normandie (1), INSA Rouen (2),
#  tutelles du laboratoire LITIS (1 et 2)
#  contributors :
#  - Denis Coquenet
#
#  This software is a computer program written in Python whose purpose is 
#  to recognize text and layout from full-page images with end-to-end deep neural networks.
#
#  This software is governed by the CeCILL-C license under French law and
#  abiding by the rules of distribution of free software.  You can  use,
#  modify and/ or redistribute the software under the terms of the CeCILL-C
#  license as circulated by CEA, CNRS and INRIA at the following URL
#  "http://www.cecill.info".
#
#  As a counterpart to the access to the source code and  rights to copy,
#  modify and redistribute granted by the license, users are provided only
#  with a limited warranty  and the software's author,  the holder of the
#  economic rights,  and the successive licensors  have only  limited
#  liability.
#
#  In this respect, the user's attention is drawn to the risks associated
#  with loading,  using,  modifying and/or developing or reproducing the
#  software by the user in light of its specific status of free software,
#  that may mean  that it is complicated to manipulate,  and  that  also
#  therefore means  that it is reserved for developers  and  experienced
#  professionals having in-depth computer knowledge. Users are therefore
#  encouraged to load and test the software's suitability as regards their
#  requirements in conditions enabling the security of their systems and/or
#  data to be ensured and,  more generally, to use and operate it in the
#  same conditions as regards security.
#
#  The fact that you are presently reading this means that you have had
#  knowledge of the CeCILL-C license and that you accept its terms.

from OCR.ocr_manager import OCRManager
from torch.nn import CrossEntropyLoss
import torch
from OCR.ocr_utils import LM_ind_to_str
import numpy as np
from torch.amp import autocast
import time
import logging
import torch.nn as nn
import os
from PIL import Image

def check_mem():
    print("Allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
    print("Reserved: ", torch.cuda.memory_reserved()  / 1024**2, "MB")

class Manager(OCRManager):

    def __init__(self, params):
        super(Manager, self).__init__(params)
        self.batchCount = 0
        self.logger = logging.getLogger("DAN_trainer")
    def load_save_info(self, info_dict):
        if "curriculum_config" in info_dict.keys():
            if self.dataset is not None and self.dataset.train_dataset is not None:
                self.dataset.train_dataset.curriculum_config = info_dict["curriculum_config"]

    def add_save_info(self, info_dict):
        info_dict["curriculum_config"] = self.dataset.train_dataset.curriculum_config
        return info_dict

    def get_init_hidden(self, batch_size):
        num_layers = 1
        hidden_size = self.params["model_params"]["enc_dim"]
        return torch.zeros(num_layers, batch_size, hidden_size), torch.zeros(num_layers, batch_size, hidden_size)

    def apply_teacher_forcing(self, y, y_len, error_rate):
        y_error = y.clone()
        for b in range(len(y_len)):
            for i in range(1, y_len[b]):
                if np.random.rand() < error_rate and y[b][i] != self.dataset.tokens["pad"]:
                    y_error[b][i] = np.random.randint(0, len(self.dataset.charset)+2)
        return y_error, y_len


    def save_batch(self, batch_data, output_dir, start_index=0, gt=None):
        """
        Saves images and matching ground truth labels.
        
        Parameters
        ----------
        batch_data: dict
            Must contain "imgs" and "gt".
        output_dir: str
            Directory to save the images + text files.
        start_index: int
            Starting index for filenames (useful if called repeatedly).
        tokens_to_text: callable
            Function that converts token IDs to strings (ground truth).
        """

        os.makedirs(output_dir, exist_ok=True)

        imgs = batch_data["imgs"]  # (B, C, H, W)
        #gts  = batch_data["gt"]    # (B, T)

        B = imgs.size(0)

        for i in range(B):
            idx = start_index + i

            # ---- Save the image ----
            img_tensor = imgs[i].detach().cpu()
            img_tensor = img_tensor * 0.5 + 0.5
            # -----------------------------------------

            img_tensor = img_tensor.clamp(0, 1)
            # convert CHW -> HWC (and scale if needed)
            if img_tensor.size(0) == 1:
                img_array = (img_tensor.squeeze(0).numpy() * 255).astype("uint8")
                img_pil = Image.fromarray(img_array, mode="L")
            else:
                img_array = (img_tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
                img_pil = Image.fromarray(img_array)

            img_filename = os.path.join(output_dir, f"img_{idx:05d}.png")
            img_pil.save(img_filename)

            # ---- Save the ground truth ----
            text = gt[i]

            txt_filename = os.path.join(output_dir, f"img_{idx:05d}.txt")
            with open(txt_filename, "w", encoding="utf-8") as f:
                f.write(text)


    def train_batch(self, batch_data, metric_names):
        loss_func = CrossEntropyLoss(ignore_index=self.dataset.tokens["pad"])
        '''
        self.save_batch(
                batch_data,
                output_dir="debug_batches",
                start_index=self.batchCount * len(batch_data["imgs"]),
                gt=batch_data["raw_labels"]  # or similar
            )
        '''
        y = batch_data["labels"].to(self.device)
        sum_loss = 0
        x = batch_data["imgs"].to(self.device)
        #x = batch_data["imgs"].to(self.device).float()
        #mx = float(x.max())
        #if mx > 2.0:           # likely 0..255
        #    x = x / 255.0
        # Use fixed stats (match pretraining!)
        # grayscale
        #x = (x - 0.5) / 0.5    # -> roughly [-1, 1]
        # or RGB (adjust if using 3ch)
        #x = (x - torch.tensor([0.5,0.5,0.5])[None,:,None,None]) / 0.5


        
        reduced_size = [s[:2] for s in batch_data["imgs_reduced_shape"]]
        y_len = batch_data["labels_len"]

        # add errors in teacher forcing
        if "teacher_forcing_error_rate" in self.params["training_params"] and self.params["training_params"]["teacher_forcing_error_rate"] is not None:
            error_rate = self.params["training_params"]["teacher_forcing_error_rate"]
            simulated_y_pred, y_len = self.apply_teacher_forcing(y, y_len, error_rate)
        elif "teacher_forcing_scheduler" in self.params["training_params"]:
            error_rate = self.params["training_params"]["teacher_forcing_scheduler"]["min_error_rate"] + min(self.latest_step, self.params["training_params"]["teacher_forcing_scheduler"]["total_num_steps"]) * (self.params["training_params"]["teacher_forcing_scheduler"]["max_error_rate"]-self.params["training_params"]["teacher_forcing_scheduler"]["min_error_rate"]) / self.params["training_params"]["teacher_forcing_scheduler"]["total_num_steps"]
            simulated_y_pred, y_len = self.apply_teacher_forcing(y, y_len, error_rate)
        else:
            simulated_y_pred = y
        num_reached_end = 0
        '''
        for name, p in self.models["encoder"].named_parameters():
            if torch.is_floating_point(p) and not torch.isfinite(p).all():
                print("BAD:", name)
        '''
        with autocast('cuda', enabled=self.params["training_params"]["use_amp"]):
            hidden_predict = None
            cache = None
            '''
            bad = {"name": None}
            def hook(name):
                def fn(_m, _inp, out):
                    t = out if torch.is_tensor(out) else out[0]
                    if not torch.isfinite(t).all() and bad["name"] is None:
                        bad["name"] = name
                return fn

            handles = []
            for n, m in self.models["encoder"].init_blocks.named_modules():
                if n:  # skip the root container
                    handles.append(m.register_forward_hook(hook(f"init_blocks.{n}")))

            raw_features = self.models["encoder"](x)  # run one batch
            print("first bad layer:", bad["name"])
            for h in handles: h.remove()
            
            enc = self.models["encoder"]
            m = enc.init_blocks[2].conv2

            # weights/bias sanity
            print("w finite?", torch.isfinite(m.weight).all().item(),
                "min/max:", float(m.weight.min()), float(m.weight.max()))
            if m.bias is not None:
                print("b finite?", torch.isfinite(m.bias).all().item(),
                    "min/max:", float(m.bias.min()), float(m.bias.max()))

            # see what goes INTO conv2
            def pre_hook(_m, inp):
                t = inp[0]
                print(">> conv2 input finite:", torch.isfinite(t).all().item(),
                    "min/max:", float(t.min()), float(t.max()))
                return None
            h = m.register_forward_pre_hook(pre_hook)

            _ = enc(x)  # run one batch
            h.remove()
            '''
            '''
            enc = self.models["encoder"]
            
            for name, module in enc.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    print(f"BN layer: {name}")
                    print(f"  num_features: {module.num_features}")
                    print(f"  running_mean (first 5): {module.running_mean[:5]}")
                    print(f"  running_var (first 5):  {module.running_var[:5]}")
                    print(f"  eps: {module.eps}")
                    print(f"  affine: {module.affine}")
                    if module.affine:
                        print(f"  weight (gamma) min/max: {module.weight.min().item():.4f}, {module.weight.max().item():.4f}")
                        print(f"  bias   (beta)  min/max: {module.bias.min().item():.4f}, {module.bias.max().item():.4f}")
                    print()
            '''

            raw_features = self.models["encoder"](x)
            #print(f'***** Encoder output x.shape={x.shape} raw_features.shape={raw_features.shape}')
            
            features_size = raw_features.size()
            b, c, h, w = features_size

            pos_features = self.models["decoder"].features_updater.get_pos_features(raw_features)
            features = torch.flatten(pos_features, start_dim=2, end_dim=3).permute(2, 0, 1)
            enhanced_features = pos_features
            enhanced_features = torch.flatten(enhanced_features, start_dim=2, end_dim=3).permute(2, 0, 1)
            output, pred, hidden_predict, cache, weights = self.models["decoder"](features, enhanced_features,
                                                                               simulated_y_pred[:, :-1],
                                                                               reduced_size,
                                                                               [max(y_len) for _ in range(b)],
                                                                               features_size,
                                                                               start=0,
                                                                               hidden_predict=hidden_predict,
                                                                               cache=cache,
                                                                               keep_all_weights=True)

            loss_ce = loss_func(pred, y[:, 1:])
            sum_loss += loss_ce
            with autocast('cuda', enabled=False):
                self.backward_loss(sum_loss)
                self.step_optimizers()
                self.zero_optimizers()
            predicted_tokens = torch.argmax(pred, dim=1).detach().cpu().numpy()
            predicted_tokens = [predicted_tokens[i, :y_len[i]] for i in range(b)]
            '''
            if any(99 in tokens for tokens in predicted_tokens):
                print("99 is in the list")
            else:
                print("99 is not in the list")
            '''
            str_x = [LM_ind_to_str(self.dataset.charset, t, oov_symbol="") for t in predicted_tokens]
            self.batchCount += 1
            if self.params["training_params"]["showGroundTruthAndPrediction"]:
                
                if self.batchCount % 1000 == 0:
                    print(f"Batch {self.batchCount} - Loss: {sum_loss.item()} - Error rate: {error_rate if 'error_rate' in locals() else 'N/A'}")
                    print(f"{batch_data["raw_labels"]} =>  {str_x}")

        values = {
            "nb_samples": b,
            "str_y": batch_data["raw_labels"],
            "str_x": str_x,
            "loss": sum_loss.item(),
            "loss_ce": loss_ce.item(),
            "syn_max_lines": self.dataset.train_dataset.get_syn_max_lines() if self.params["dataset_params"]["config"]["synthetic_data"] else 0,
        }
        # check_mem()
        return values
        '''
        def has_repeated_tail(tokens, repeat_threshold=5):
            if len(tokens) < repeat_threshold:
                return False
            return all(t == tokens[-1] for t in tokens[-repeat_threshold:])
        def has_ngram_repetition(tokens, n=4, repeat_times=2):
            if len(tokens) < n * repeat_times:
                return False
            recent_ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
            count = {}
            for ngram in recent_ngrams:
                count[ngram] = count.get(ngram, 0) + 1
                if count[ngram] >= repeat_times:
                    return True
            return False
        def has_ngram_repetition_at_end(tokens, n=4, repeat_times=2):
            if len(tokens) < n * repeat_times:
                return False
            tail = tokens[-n * repeat_times:]
            ngrams = [tuple(tail[i * n:(i + 1) * n]) for i in range(repeat_times)]
            return all(ngram == ngrams[0] for ngram in ngrams)
        '''
    '''
    def evaluate_batch(self, batch_data, metric_names):
        start_time = time.time()
        max_chars = self.params["training_params"]["max_char_prediction"]
        x = batch_data["imgs"].to(self.device)
        b = x.size(0)
        #b = features_size[0] if isinstance(features_size, torch.Size) else features.size(1)
        end_id = self.dataset.tokens["end"]
        pad_id = self.dataset.tokens.get("pad", 0)
        use_conf = True #False  # set True only when you actually need confidences

        # Preallocate
        pred_tokens = torch.full((b, max_chars), pad_id, device=self.device, dtype=torch.long)
        pred_len    = torch.zeros(b, device=self.device, dtype=torch.int32)
        reached_end = torch.zeros(b, device=self.device, dtype=torch.bool)

        # Optional: per-step confidence (avoid unless needed)
        if use_conf:
            conf_mat = torch.empty((b, max_chars), device=self.device, dtype=torch.float32)

        # Decoder state
        cache = None
        hidden_predict = None
        # coverage vector & weights only if you use them later
        coverage_vector = None

        # To start, decoder expects previous tokens with BOS at t=0
        prev_tokens = predicted_tokens  # your existing BOS tensor of shape [B, 1]
        prev_len    = predicted_tokens_len  # [B]

        for t in range(max_chars):
            # Narrow to active samples to avoid work on finished sequences
            active = (~reached_end).nonzero(as_tuple=False).squeeze(1)
            if active.numel() == 0:
                break

            # Views of active items
            prev_tokens_a = prev_tokens.index_select(0, active)
            prev_len_a    = prev_len.index_select(0, active)

            # If your decoder supports cached state per sample, you may also need
            # to index_select cache/hidden_predict here to keep them aligned with `active`.

            output, pred, hidden_predict, cache, weights = self.models["decoder"](
                features, enhanced_features, prev_tokens_a, reduced_size, prev_len_a,
                features_size, start=0, hidden_predict=hidden_predict, cache=cache, num_pred=1
            )

            # Only the last step’s logits are needed
            logits_last = pred[:, :, -1]           # [A, V]
            next_ids    = logits_last.argmax(dim=1)  # [A]

            # Write into the preallocated arrays at time t
            pred_tokens.index_copy_(0, active, torch.where(
                reached_end.index_select(0, active),  # should all be False by definition
                pred_tokens.index_select(0, active)[:, t],  # no-op
                next_ids
            ).unsqueeze(1).expand(-1, 1))  # write column t
            pred_tokens[active, t] = next_ids

            # Optional confidence (use log_softmax→exp for stability; skip if not needed)
            if use_conf:
                conf = logits_last.log_softmax(dim=1).amax(dim=1).exp()  # [A]
                conf_mat[active, t] = conf

            # Update reached_end & lengths (only first time we see <eos>)
            hit_eos = (next_ids == end_id)
            # set length for items that just finished at t (i.e., were active and now hit eos)
            finished_now_idx_in_active = hit_eos.nonzero(as_tuple=False).squeeze(1)
            if finished_now_idx_in_active.numel() > 0:
                finished_global = active.index_select(0, finished_now_idx_in_active)
                pred_len[finished_global] = t + 1
                reached_end[finished_global] = True

            # (Optional) coverage update only for active items
            # if you need it later; otherwise omit for speed
            # if coverage_vector is not None:
            #     coverage_vector[active] = torch.clamp(coverage_vector[active] + weights, 0, 1)

            # Prepare tokens for next step: append next_ids to prev_tokens **once** (avoid cat chains)
            # Keep a running buffer by growing columns lazily:
            prev_tokens = torch.cat([prev_tokens, next_ids.new_empty((b, 1)).fill_(pad_id)], dim=1)
            prev_tokens[active, -1] = next_ids
            prev_len[active] += 1

            # Early stop if all finished
            if reached_end.all():
                break

        # Finalize lengths for those that never hit <eos>
        pred_len[~reached_end] = max_chars

        # Slice per sample outputs to their lengths
        predicted_tokens = [pred_tokens[i, :pred_len[i].item()] for i in range(b)]
        if use_conf:
            confidence_scores = [conf_mat[i, :pred_len[i].item()].tolist() for i in range(b)]
        else:
            confidence_scores = None

        process_time = time.time() - start_time



        confidence_scores = torch.cat(confidence_scores, dim=1).cpu().detach().numpy()
        predicted_tokens = predicted_tokens[:, 1:]
        prediction_len[torch.eq(reached_end, False)] = max_chars - 1
        predicted_tokens = [predicted_tokens[i, :prediction_len[i]] for i in range(b)]
        confidence_scores = [confidence_scores[i, :prediction_len[i]].tolist() for i in range(b)]
        str_x = [LM_ind_to_str(self.dataset.charset, t, oov_symbol="") for t in predicted_tokens]
 

        process_time = time.time() - start_time
        
        values = {
            "nb_samples": b,
            "str_y": batch_data["raw_labels"],
            "str_x": str_x,
            "confidence_score": confidence_scores,
            "time": process_time,
        }
        return values
    '''

    def evaluate_batch(self, batch_data, metric_names):

        #print(f"Processing file {batch_data['names'][0]} with {len(batch_data['names'])} samples")
        x = batch_data["imgs"].to(self.device)
        reduced_size = [s[:2] for s in batch_data["imgs_reduced_shape"]]

        max_chars = self.params["training_params"]["max_char_prediction"]

        start_time = time.time()
        with autocast('cuda', enabled=self.params["training_params"]["use_amp"]):
            b = x.size(0)
            reached_end = torch.zeros((b, ), dtype=torch.bool, device=self.device)
            prediction_len = torch.zeros((b, ), dtype=torch.int, device=self.device)
            predicted_tokens = torch.ones((b, 1), dtype=torch.long, device=self.device) * self.dataset.tokens["start"]
            predicted_tokens_len = torch.ones((b, ), dtype=torch.int, device=self.device)

            ###whole_output = list()
            confidence_scores = list()
            cache = None
            hidden_predict = None
            if b > 1:
                features_list = list()
                for i in range(b):
                    pos = batch_data["imgs_position"]
                    features_list.append(self.models["encoder"](x[i:i+1, :, pos[i][0][0]:pos[i][0][1], pos[i][1][0]:pos[i][1][1]]))
                max_height = max([f.size(2) for f in features_list])
                max_width = max([f.size(3) for f in features_list])
                features = torch.zeros((b, features_list[0].size(1), max_height, max_width), device=self.device, dtype=features_list[0].dtype)
                for i in range(b):
                    features[i, :, :features_list[i].size(2), :features_list[i].size(3)] = features_list[i]
            else:
                features = self.models["encoder"](x)
            features_size = features.size()
            ###coverage_vector = torch.zeros((features.size(0), 1, features.size(2), features.size(3)), device=self.device)
            pos_features = self.models["decoder"].features_updater.get_pos_features(features)
            features = torch.flatten(pos_features, start_dim=2, end_dim=3).permute(2, 0, 1)
            enhanced_features = pos_features
            enhanced_features = torch.flatten(enhanced_features, start_dim=2, end_dim=3).permute(2, 0, 1)
            n_reached_end = 0
            
            for i in range(0, max_chars):
                #self.logger.debug(f"Step {i+1}/{max_chars}")
                output, pred, hidden_predict, cache, weights = self.models["decoder"](features, enhanced_features, 
                                                                                      predicted_tokens, 
                                                                                      reduced_size, 
                                                                                      predicted_tokens_len, 
                                                                                      features_size, start=0, 
                                                                                      hidden_predict=hidden_predict, 
                                                                                      cache=cache, num_pred=1)
                #self.logger.debug(f"Output shape: {output.shape}, Pred shape: {pred.shape}, Hidden shape: {hidden_predict.shape if hidden_predict is not None else 'N/A'}, Cache shape: {cache.shape if cache is not None else 'N/A'}")
                ###whole_output.append(output)
                #self.logger.debug(f"Predicted tokens shape: {predicted_tokens.shape}, Predicted tokens len: {predicted_tokens_len}")
                confidence_scores.append(torch.max(torch.softmax(pred[:, :], dim=1), dim=1).values)
                # pred: (B, V, t)
                #last_step = pred[:, :, -1]                                # (B, V)
                #last_conf = torch.softmax(last_step, dim=1).max(dim=1).values  # (B,)
                #confidence_scores.append(last_conf)                       # append a (B,) tensor

                #self.logger.debug(f"Confidence scores shape: {confidence_scores[-1].shape}")
                ###coverage_vector = torch.clamp(coverage_vector + weights, 0, 1)
                #self.logger.debug(f"Coverage vector shape: {coverage_vector.shape}")
                predicted_tokens = torch.cat([predicted_tokens, torch.argmax(pred[:, :, -1], dim=1, keepdim=True)], dim=1)
                #self.logger.debug(f"Predicted tokens after cat: {predicted_tokens.shape}")
                reached_end = torch.logical_or(reached_end, torch.eq(predicted_tokens[:, -1], self.dataset.tokens["end"]))
                #self.logger.debug(f"Reached end shape: {reached_end.shape}, N reached end: {n_reached_end}")
                # Count how many elements are True in reached_end
                if self.params["training_params"]["showGroundTruthAndPrediction"]:
                    num_reached_end = reached_end.sum().item()
                    if n_reached_end != num_reached_end:
                        n_reached_end = num_reached_end
                        print(f"Reached end: {n_reached_end}")
                predicted_tokens_len += 1

                prediction_len[reached_end == False] = i + 1
                if torch.all(reached_end):
                    break
                #self.logger.debug(f"Prediction length: {prediction_len}")
            '''
            B = x.size(0)
            T = max_chars
            pred_ids = torch.full((B, T+1), self.dataset.tokens["start"], device=self.device, dtype=torch.long)
            conf_mat = torch.empty((B, T), device=self.device, dtype=torch.float32)
            reached_end = torch.zeros((B,), dtype=torch.bool, device=self.device)

            for i in range(T):
                output, pred, hidden_predict, cache, weights = self.models["decoder"](features, enhanced_features, 
                                                                                      predicted_tokens, 
                                                                                      reduced_size, 
                                                                                      predicted_tokens_len, 
                                                                                      features_size, start=0, 
                                                                                      hidden_predict=hidden_predict, 
                                                                                      cache=cache, num_pred=1)

                

                last_step = pred[:, :, -1]
                next_tok = last_step.argmax(dim=1)            # (B,)
                pred_ids[:, i+1] = next_tok                   # in-place write

                last_conf = torch.softmax(last_step, dim=1).max(dim=1).values  # (B,)
                confidence_scores.append(last_conf.unsqueeze(1)) 
                conf_mat[:, i] = last_conf

                #conf_mat[:, i] = torch.softmax(last_step, 1).amax(1).values
                reached_end |= next_tok.eq(self.dataset.tokens["end"])
                if torch.all(reached_end):
                    final_len = i + 1
                    break
            else:
                final_len = T

            # slice to actual decoded lengths later when you build strings
            '''
            confidence_scores = torch.cat(confidence_scores, dim=1).cpu().detach().numpy()
            # from list of (B,) → (B, T_decoded)
            #confidence_scores = torch.stack(confidence_scores, dim=1).cpu().detach().numpy()
            predicted_tokens = predicted_tokens[:, 1:]
            prediction_len[torch.eq(reached_end, False)] = max_chars - 1
            predicted_tokens = [predicted_tokens[i, :prediction_len[i]] for i in range(b)]
            confidence_scores = [confidence_scores[i, :prediction_len[i]].tolist() for i in range(b)]
            str_x = [LM_ind_to_str(self.dataset.charset, t, oov_symbol="") for t in predicted_tokens]
            if self.params["training_params"]["showGroundTruthAndPrediction"]:
                print(f"{batch_data["raw_labels"]} =>  {str_x}")

        process_time = time.time() - start_time
        
        values = {
            "nb_samples": b,
            "str_y": batch_data["raw_labels"],
            "str_x": str_x,
            "confidence_score": confidence_scores,
            "time": process_time,
            
        }
        if self.params["training_params"]["showGroundTruthAndPrediction"]:
            print(f"{process_time} seconds for {b} samples")

        return values
