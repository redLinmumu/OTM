
"""Implements supervised learning training procedures."""
import torch
from torch import nn
import time
from eval_scripts.performance import AUPRC, f1_score, accuracy, eval_affect
from eval_scripts.complexity import all_in_one_train, all_in_one_test
from eval_scripts.robustness import relative_robustness, effective_robustness, single_plot
from tqdm import tqdm
from mwae.mixer import mixer_aug, mixer_aug_regression, mixer_fuse
from mwae.get_distance import distance_tensor
import os
import numpy as np
import matplotlib.pyplot as plt

softmax = nn.Softmax()


class MMDL(nn.Module):
    """Implements MMDL classifier."""
    
    def __init__(self, encoders, fusion, head, has_padding=False):
        """Instantiate MMDL Module

        Args:
            encoders (List): List of nn.Module encoders, one per modality.
            fusion (nn.Module): Fusion module
            head (nn.Module): Classifier module
            has_padding (bool, optional): Whether input has padding or not. Defaults to False.
        """
        super(MMDL, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.fuse = fusion
        self.head = head
        self.has_padding = has_padding
        self.fuseout = None
        self.reps = []

    def forward(self, inputs):
        """Apply MMDL to Layer Input.

        Args:
            inputs (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        outs = []
        if self.has_padding:
            for i in range(len(inputs[0])):
                outs.append(self.encoders[i](
                    [inputs[0][i], inputs[1][i]]))
        else:
            for i in range(len(inputs)):
                outs.append(self.encoders[i](inputs[i]))
        self.reps = outs
        if self.has_padding:
            
            if isinstance(outs[0], torch.Tensor):
                out = self.fuse(outs)
            else:
                out = self.fuse([i[0] for i in outs])
        else:
            out = self.fuse(outs)
        self.fuseout = out
        if type(out) is tuple:
            out = out[0]
        if self.has_padding and not isinstance(outs[0], torch.Tensor):
            return self.head([out, inputs[1][0]])
        return self.head(out)
    
    def encode(self, inputs):
        outs = []
        if self.has_padding:
            for i in range(len(inputs[0])):
                outs.append(self.encoders[i](
                    [inputs[0][i], inputs[1][i]]))
        else:
            for i in range(len(inputs)):
                outs.append(self.encoders[i](inputs[i]))
        self.reps = outs
        return outs

    def fuse_head(self, outs):
        if self.has_padding:
            
            if isinstance(outs[0], torch.Tensor):
                out = self.fuse(outs)
            else:
                out = self.fuse([i[0] for i in outs])
        else:
            out = self.fuse(outs)
        self.fuseout = out
        if type(out) is tuple:
            out = out[0]
        if self.has_padding and not isinstance(outs[0], torch.Tensor):
            return self.head([out, outs[1][0]])
        return self.head(out)



class SingleMDL(nn.Module):
    """Implements single MMDL classifier."""
    
    def __init__(self, encoder, head, index=-1, has_padding=False, fuse_need=None):
        """Instantiate single  MMDL Module

        Args:
            encoders (List): List of nn.Module encoders, one per modality.
            head (nn.Module): Classifier module
            has_padding (bool, optional): Whether input has padding or not. Defaults to False.
        """
        super(SingleMDL, self).__init__()
        self.encoder = encoder
        self.head = head
        self.has_padding = has_padding
        self.index = index
        self.fuse_need = fuse_need
        self.reps = None
        self.fuseout = None

    def forward(self, input):
        if self.has_padding:  
            input_i = input[0][self.index]
            label_i = input[1][self.index]
            out = self.encoder([input_i, label_i])
        else:
            out = self.encoder(input)
            self.reps = out
            if self.fuse_need is not None:
                out = self.fuse_need(out)
                self.fuseout = out
        output = self.head(out)
        return output
    

def deal_with_objective(objective, pred, truth, args):
    """Alter inputs depending on objective function, to deal with different objective arguments."""
    
    try:
        if args is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # del args['device'] 
        else:
            device = args['device']
    except:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    if type(objective) == nn.CrossEntropyLoss:
        if len(truth.size()) == len(pred.size()):
            truth1 = truth.squeeze(len(pred.size())-1)
        else: 
            truth1 = truth
        return objective(pred, truth1.long().to(device))
    elif type(objective) == nn.MSELoss or type(objective) == nn.modules.loss.BCEWithLogitsLoss or type(objective) == nn.L1Loss:        
        return objective(pred, truth.float().to(device))
    else:
        return objective(pred, truth.to(device), args)


def train(
        encoders, fusion, head, train_dataloader, valid_dataloader, total_epochs, additional_optimizing_modules=[], is_packed=False,
        early_stop=False, task="classification", optimtype=torch.optim.RMSprop, lr=0.001, weight_decay=0.0,
        objective=nn.CrossEntropyLoss(), auprc=False, save='best.pt', validtime=False, objective_args_dict=None, input_to_float=True, clip_val=8,
        track_complexity=True, device=None, is_load=False, tolerance=7):
    """
    Handle running a simple supervised training loop.
    
    :param encoders: list of modules, unimodal encoders for each input modality in the order of the modality input data.
    :param fusion: fusion module, takes in outputs of encoders in a list and outputs fused representation
    :param head: classification or prediction head, takes in output of fusion module and outputs the classification or prediction results that will be sent to the objective function for loss calculation
    :param total_epochs: maximum number of epochs to train
    :param additional_optimizing_modules: list of modules, include all modules that you want to be optimized by the optimizer other than those in encoders, fusion, head (for example, decoders in MVAE)
    :param is_packed: whether the input modalities are packed in one list or not (default is False, which means we expect input of [tensor(20xmodal1_size),(20xmodal2_size),(20xlabel_size)] for batch size 20 and 2 input modalities)
    :param early_stop: whether to stop early if valid performance does not improve over 7 epochs
    :param task: type of task, currently support "classification","regression","multilabel"
    :param optimtype: type of optimizer to use
    :param lr: learning rate
    :param weight_decay: weight decay of optimizer
    :param objective: objective function, which is either one of CrossEntropyLoss, MSELoss or BCEWithLogitsLoss or a custom objective function that takes in three arguments: prediction, ground truth, and an argument dictionary.
    :param auprc: whether to compute auprc score or not
    :param save: the name of the saved file for the model with current best validation performance
    :param validtime: whether to show valid time in seconds or not
    :param objective_args_dict: the argument dictionary to be passed into objective function. If not None, at every batch the dict's "reps", "fused", "inputs", "training" fields will be updated to the batch's encoder outputs, fusion module output, input tensors, and boolean of whether this is training or validation, respectively.
    :param input_to_float: whether to convert input to float type or not
    :param clip_val: grad clipping limit
    :param track_complexity: whether to track training complexity or not
    """
    if device is None:
        
        device_str = "cuda:0"

        device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    
    print(device)

    model = MMDL(encoders, fusion, head, has_padding=is_packed).to(device)
    
    if is_load:
        if os.path.exists(save):
            model = torch.load(save)
            print("Load saved model <Origin>")
        
    def _trainprocess():
        additional_params = []
        for m in additional_optimizing_modules:
            additional_params.extend(
                [p for p in m.parameters() if p.requires_grad])
        op = optimtype([p for p in model.parameters() if p.requires_grad] +
                       additional_params, lr=lr, weight_decay=weight_decay)
        bestvalloss = 10000
        bestacc = 0
        bestf1 = 0
        patience = 0

        def _processinput(inp):
            if input_to_float:
                return inp.float()
            else:
                return inp

        for epoch in range(total_epochs):
            totalloss = 0.0
            totals = 0
            model.train()
            for j in train_dataloader:
                op.zero_grad()
                if is_packed:
                    with torch.backends.cudnn.flags(enabled=False):
                        model.train()
                        out = model([[_processinput(i).to(device)
                                    for i in j[0]], j[1]])
                else:
                    model.train()
                    out = model([_processinput(i).to(device)
                                for i in j[:-1]])
                if not (objective_args_dict is None):
                    objective_args_dict['reps'] = model.reps
                    objective_args_dict['fused'] = model.fuseout
                    objective_args_dict['inputs'] = j[:-1]
                    objective_args_dict['training'] = True
                    objective_args_dict['model'] = model
                    objective_args_dict['device']=device
                    loss = deal_with_objective(
                        objective, out, j[-1], objective_args_dict)
                else:
                    args = {}
                    args['device'] = device
                    loss = deal_with_objective(
                        objective, out, j[-1], args)

                totalloss += loss * len(j[-1])
                totals += len(j[-1])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                op.step()
            print("Epoch "+str(epoch)+" train loss: "+str((totalloss/totals).item()))
            validstarttime = time.time()
            if validtime:
                print("train total: "+str(totals))
            model.eval()
            with torch.no_grad():
                totalloss = 0.0
                pred = []
                true = []
                pts = []
                for j in valid_dataloader:
                    if is_packed:
                        out = model([[_processinput(i).to(device)
                                    for i in j[0]], j[1]])
                    else:
                        out = model([_processinput(i).to(device)
                                    for i in j[:-1]])

                    if not (objective_args_dict is None):
                        objective_args_dict['reps'] = model.reps
                        objective_args_dict['fused'] = model.fuseout
                        objective_args_dict['inputs'] = j[:-1]
                        objective_args_dict['training'] = False
                        objective_args_dict['device']=device
                        loss = deal_with_objective(
                            objective, out, j[-1], objective_args_dict)
                    else:
                        args = {}
                        args['device'] = device
                        loss = deal_with_objective(
                            objective, out, j[-1], args)
                        
                    totalloss += loss*len(j[-1])
                    
                    if task == "classification":
                        pred.append(torch.argmax(out, 1))
                    elif task == "multilabel":
                        pred.append(torch.sigmoid(out).round())
                    true.append(j[-1])
                    if auprc:
                        # pdb.set_trace()
                        sm = softmax(out)
                        pts += [(sm[i][1].item(), j[-1][i].item())
                                for i in range(j[-1].size(0))]
            if pred:
                pred = torch.cat(pred, 0)
            true = torch.cat(true, 0)
            totals = true.shape[0]
            valloss = totalloss/totals
            if task == "classification":
                acc = accuracy(true, pred)
                print("Epoch "+str(epoch)+" valid loss: "+str(valloss.item()) +
                      " acc: "+str(acc))
                if acc > bestacc:
                    patience = 0
                    bestacc = acc
                    print("Saving Best")
                    torch.save(model, save)
                 
                    len_additional = len(additional_optimizing_modules)
                    if len_additional > 0:
                        save_list_dir = save[:-3]
                        for indx in range(len_additional):
                            save_indx_dir = save_list_dir + '_add_' + str(indx) + 'pt'
                            torch.save(additional_optimizing_modules[indx], save_indx_dir)
                else:
                    patience += 1
            elif task == "multilabel":
                f1_micro = f1_score(true, pred, average="micro")
                f1_macro = f1_score(true, pred, average="macro")
                print("Epoch "+str(epoch)+" valid loss: "+str(valloss) +
                      " f1_micro: "+str(f1_micro)+" f1_macro: "+str(f1_macro))
                if f1_macro > bestf1:
                    patience = 0
                    bestf1 = f1_macro
                    print("Saving Best")
                    torch.save(model, save)
                else:
                    patience += 1
            elif task == "regression":
                print("Epoch "+str(epoch)+" valid loss: "+str(valloss.item()))
                if valloss < bestvalloss:
                    patience = 0
                    bestvalloss = valloss
                    print("Saving Best")
                    torch.save(model, save)
                else:
                    patience += 1
            if early_stop and patience > tolerance:
                print(f'early_stop and patience > {tolerance}')
                break
            if auprc:
                print("AUPRC: "+str(AUPRC(pts)))
            validendtime = time.time()
            if validtime:
                print("valid time:  "+str(validendtime-validstarttime))
                print("Valid total: "+str(totals))
        
        if task == "classification":
            final_best = bestacc
        elif task == 'regression':
            final_best = bestvalloss
        elif task == 'multilabel':
            final_best = bestf1
        print(f'Final best = {final_best}')
        return final_best
    
    track_complexity = False
    if track_complexity:
        all_in_one_train(_trainprocess, [model]+additional_optimizing_modules)
    else:
        return _trainprocess()


def train_mixer(encoders, fusion, head, train_dataloader, valid_dataloader, total_epochs, additional_optimizing_modules=[], is_packed=False,
        early_stop=False, task="classification", optimtype=torch.optim.RMSprop, lr=0.001, weight_decay=0.0,
        objective=nn.CrossEntropyLoss(), auprc=False, save='best.pt', validtime=False, objective_args_dict=None, input_to_float=True, clip_val=8,
        track_complexity=True, device=None, is_load=False, beta=1.0, tolerance=7):
    
    print(f'beta = {beta} is the weight of loss_mixer.')
    print(f'task = {task}')
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    # torch.autograd.set_detect_anomaly(True)
    # print(f">>> Device = {device}")

    model = MMDL(encoders, fusion, head, has_padding=is_packed).to(device)

    if is_load:
        if os.path.exists(save):
            model = torch.load(save)
            print("Load saved model <Mixer>")

    def _trainprocess():
        additional_params = []
        for m in additional_optimizing_modules:
            additional_params.extend(
                [p for p in m.parameters() if p.requires_grad])
        op = optimtype([p for p in model.parameters() if p.requires_grad] +
                       additional_params, lr=lr, weight_decay=weight_decay)
        bestvalloss = 10000
        bestacc = 0
        bestf1 = 0
        patience = 0

        def _processinput(inp):
            if input_to_float:
                return inp.float()
            else:
                return inp
            
        
        for epoch in range(total_epochs):
            totalloss = 0.0
            totals = 0
            model.train()
            for j in train_dataloader:
                op.zero_grad()
                if is_packed:
                    with torch.backends.cudnn.flags(enabled=False):
                        model.train()
                        out = model([[_processinput(i).to(device)
                                    for i in j[0]], j[1]])
                else:
                    model.train()
                    out = model([_processinput(i).to(device)
                                for i in j[:-1]])
                                  
                
                if not (objective_args_dict is None):
                    objective_args_dict['reps'] = model.reps
                    objective_args_dict['fused'] = model.fuseout
                    objective_args_dict['inputs'] = j[:-1]
                    objective_args_dict['training'] = True
                    objective_args_dict['model'] = model
                    objective_args_dict['device'] = device

                    loss = deal_with_objective(
                    objective, out, j[-1], objective_args_dict)
                else:
                    args = {}
                    args['device'] = device
                    loss = deal_with_objective(
                        objective, out, j[-1], args)

                if is_packed:
                    # import os

                 
                    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
                    zs = model.encode([[_processinput(i).to(device)  for i in j[0]], j[1]])
                        
                else:
                    zs = model.encode([_processinput(i).to(device) for i in j[:-1]])
                
                if task == 'classification':
                    zs_aug, labels_aug = mixer_aug(zs, j[-1])
                elif task == 'regression':
                    zs_aug, labels_aug = mixer_aug_regression(zs, j[-1].to(device))
                    
                # print(f"labels_aug")
                out_mixer = model.fuse_head(zs_aug)
                if objective_args_dict is None:
                    loss_mixer = deal_with_objective(objective, out_mixer, labels_aug, args)
                else:
                    loss_mixer = deal_with_objective(objective, out_mixer, labels_aug, objective_args_dict)
                
                # print(f'Epoch [{epoch}]: loss_origin={loss}, loss_mixer={loss_mixer}')
                
                loss += beta * loss_mixer

                totalloss += loss * len(j[-1])
                totals += len(j[-1])
                
                loss.backward()
                # print('>>>loss.backward()')
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                op.step()
            print("Epoch "+str(epoch)+" train loss: "+str((totalloss/totals).item()))
            validstarttime = time.time()
            if validtime:
                print("train total: "+str(totals))
            model.eval()
            with torch.no_grad():
                totalloss = 0.0
                pred = []
                true = []
                pts = []
                for j in valid_dataloader:
                    if is_packed:
                        out = model([[_processinput(i).to(device)
                                    for i in j[0]], j[1]])
                    else:
                        out = model([_processinput(i).to(device)
                                    for i in j[:-1]])

                    if not (objective_args_dict is None):
                        objective_args_dict['reps'] = model.reps
                        objective_args_dict['fused'] = model.fuseout
                        objective_args_dict['inputs'] = j[:-1]
                        objective_args_dict['training'] = False
                        objective_args_dict['device'] = device
                        loss = deal_with_objective(
                        objective, out, j[-1], objective_args_dict)
                    else:
                        args = {}
                        args['device'] = device
                        loss = deal_with_objective(objective, out, j[-1], args)
                    totalloss += loss*len(j[-1])
                    
                    if task == "classification":
                        pred.append(torch.argmax(out, 1))
                    elif task == "multilabel":
                        pred.append(torch.sigmoid(out).round())
                    true.append(j[-1])
                    if auprc:
                        # pdb.set_trace()
                        sm = softmax(out)
                        pts += [(sm[i][1].item(), j[-1][i].item())
                                for i in range(j[-1].size(0))]
            if pred:
                pred = torch.cat(pred, 0)
            true = torch.cat(true, 0)
            totals = true.shape[0]
            valloss = totalloss/totals
            if task == "classification":
                acc = accuracy(true, pred)
                print("Epoch "+str(epoch)+" valid loss: "+str(valloss.item()) +
                      " acc: "+str(acc))
                if acc > bestacc:
                    patience = 0
                    bestacc = acc
                    print("Saving Best")
                    torch.save(model, save)
                    
                    len_additional = len(additional_optimizing_modules)
                    if len_additional > 0:
                        save_list_dir = save[:-3]
                        for indx in range(len_additional):
                            save_indx_dir = save_list_dir + '_add_' + str(indx) + 'pt'
                            torch.save(additional_optimizing_modules[indx], save_indx_dir)
                else:
                    patience += 1
            elif task == "multilabel":
                f1_micro = f1_score(true, pred, average="micro")
                f1_macro = f1_score(true, pred, average="macro")
                print("Epoch "+str(epoch)+" valid loss: "+str(valloss) +
                      " f1_micro: "+str(f1_micro)+" f1_macro: "+str(f1_macro))
                if f1_macro > bestf1:
                    patience = 0
                    bestf1 = f1_macro
                    print("Saving Best")
                    torch.save(model, save)
                else:
                    patience += 1
            elif task == "regression":
                print("Epoch "+str(epoch)+" valid loss: "+str(valloss.item()))
                if valloss < bestvalloss:
                    patience = 0
                    bestvalloss = valloss
                    print("Saving Best")
                    torch.save(model, save)
                else:
                    patience += 1
            if early_stop and patience > tolerance:
                print(f'early_stop and patience > {tolerance}')
                break
            if auprc:
                print("AUPRC: "+str(AUPRC(pts)))
            validendtime = time.time()
            if validtime:
                print("valid time:  "+str(validendtime-validstarttime))
                print("Valid total: "+str(totals))
            
        if task == "classification":
            final_best = bestacc
        elif task == 'regression':
            final_best = bestvalloss
        elif task == 'multilabel':
            final_best = bestf1
        print(f'Final best = {final_best}')
        return final_best
        
    track_complexity = False
    if track_complexity:
        return all_in_one_train(_trainprocess, [model]+additional_optimizing_modules)
    else:
        return _trainprocess()
    
def train_single_modality(encoder, index, head, train_dataloader, valid_dataloader, total_epochs, additional_optimizing_modules=[], is_packed=False,
        early_stop=False, task="classification", optimtype=torch.optim.RMSprop, lr=0.001, weight_decay=0.0,
        objective=nn.CrossEntropyLoss(), auprc=False, save='best.pt', validtime=False, objective_args_dict=None, input_to_float=True, clip_val=8,
        track_complexity=True, device=None, is_load=False, beta=1.0, tolerance=7, fold_index=-1, fuse_need=None, type=None):

 
    print(f'task = {task}')
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


    model = SingleMDL(encoder, head=head, index=index, has_padding=is_packed, fuse_need=fuse_need).to(device)
    print(f'is_packed = {is_packed}')
    # if is_load:
    #     if os.path.exists(save):
    #         model = torch.load(save)
    #         print("Load saved model <Mixer>")

    def _trainprocess():
        additional_params = []
        for m in additional_optimizing_modules:
            additional_params.extend(
                [p for p in m.parameters() if p.requires_grad])
        op = optimtype([p for p in model.parameters() if p.requires_grad] +
                       additional_params, lr=lr, weight_decay=weight_decay)
        bestvalloss = 10000
        bestacc = 0
        bestf1 = 0
        patience = 0

        def _processinput(inp):
            if input_to_float:
                return inp.float()
            else:
                return inp
        if type is None:    
            save_path = save + "single_view_" + str(index) + "_fold_" + str(fold_index) + ".pt"
        else:
            save_path = save + "single_view_" + str(index) + "_fold_" + str(fold_index) + '_' + type + ".pt"
            
        for epoch in range(total_epochs):
            totalloss = 0.0
            totals = 0
            model.train()
            for j in train_dataloader:
                op.zero_grad()
                if is_packed:
                    with torch.backends.cudnn.flags(enabled=False):
                        model.train()
                        out = model([[_processinput(i).to(device)
                                    for i in j[0]], j[1]])
                else:
                    model.train()
                    input = j[index]
                    out = model(_processinput(input).to(device))
                    # labels = j[-1]
                if not (objective_args_dict is None):
                    objective_args_dict['inputs'] = j[:-1]
                    objective_args_dict['training'] = True
                    objective_args_dict['model'] = model
                    objective_args_dict['device'] = device
                    objective_args_dict['reps'] = model.reps
                    objective_args_dict['fused'] = model.fuseout
                    loss = deal_with_objective(
                    objective, out, j[-1], objective_args_dict)
                else:
                    args = {}
                    args['device'] = device
                    loss = deal_with_objective(
                        objective, out, j[-1], args)
            

                totalloss += loss * len(j[-1])
                totals += len(j[-1])
                
                loss.backward()
                # print('>>>loss.backward()')
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                op.step()
            print("Epoch "+str(epoch)+" train loss: "+str((totalloss/totals).item()))
            validstarttime = time.time()
            if validtime:
                print("train total: "+str(totals))
            model.eval()
            with torch.no_grad():
                totalloss = 0.0
                pred = []
                true = []
                pts = []
                for j in valid_dataloader:
                    if is_packed:
                        out = model([[_processinput(i).to(device)
                                    for i in j[0]], j[1]])
                    else:
                        input = j[index]
                        out = model(_processinput(input).to(device))

                    if not (objective_args_dict is None):
                        objective_args_dict['reps'] = model.reps
                        objective_args_dict['fused'] = model.fuseout
                        objective_args_dict['inputs'] = j[:-1]
                        objective_args_dict['training'] = False
                        objective_args_dict['device'] = device
                        loss = deal_with_objective(
                        objective, out, j[-1], objective_args_dict)
                    else:
                        args = {}
                        args['device'] = device
                        loss = deal_with_objective(objective, out, j[-1], args)
                    totalloss += loss*len(j[-1])
                    
                    if task == "classification":
                        pred.append(torch.argmax(out, 1))
                    elif task == "multilabel":
                        pred.append(torch.sigmoid(out).round())
                    true.append(j[-1])
                    if auprc:
                        # pdb.set_trace()
                        sm = softmax(out)
                        pts += [(sm[i][1].item(), j[-1][i].item())
                                for i in range(j[-1].size(0))]
            if pred:
                pred = torch.cat(pred, 0)
            true = torch.cat(true, 0)
            totals = true.shape[0]
            valloss = totalloss/totals
            if task == "classification":
                acc = accuracy(true, pred)
                print("Epoch "+str(epoch)+" valid loss: "+str(valloss.item()) +
                      " acc: "+str(acc))
                if acc > bestacc:
                    patience = 0
                    bestacc = acc
                    print("Saving Best")
                    torch.save(model, save_path)
                else:
                    patience += 1
            elif task == "multilabel":
                f1_micro = f1_score(true, pred, average="micro")
                f1_macro = f1_score(true, pred, average="macro")
                print("Epoch "+str(epoch)+" valid loss: "+str(valloss) +
                      " f1_micro: "+str(f1_micro)+" f1_macro: "+str(f1_macro))
                if f1_macro > bestf1:
                    patience = 0
                    bestf1 = f1_macro
                    print("Saving Best")
                    torch.save(model, save)
                else:
                    patience += 1
            elif task == "regression":
                print("Epoch "+str(epoch)+" valid loss: "+str(valloss.item()))
                if valloss < bestvalloss:
                    patience = 0
                    bestvalloss = valloss
                    print("Saving Best")
                    torch.save(model, save_path)
                else:
                    patience += 1
            if early_stop and patience > tolerance:
                print(f'early_stop and patience > {tolerance}')
                break
            if auprc:
                print("AUPRC: "+str(AUPRC(pts)))
            validendtime = time.time()
            if validtime:
                print("valid time:  "+str(validendtime-validstarttime))
                print("Valid total: "+str(totals))
            
        if task == "classification":
            final_best = bestacc
        elif task == 'regression':
            final_best = bestvalloss
        elif task == 'multilabel':
            final_best = bestf1
        print(f'Final best = {final_best}')
        return final_best
        
    track_complexity = False
    if track_complexity:
        return all_in_one_train(_trainprocess, [model]+additional_optimizing_modules)
    else:
        return _trainprocess()
  
def train_head_single_modality(encoder, index, head, train_dataloader, valid_dataloader, total_epochs, additional_optimizing_modules=[], is_packed=False,
        early_stop=False, task="classification", optimtype=torch.optim.RMSprop, lr=0.001, weight_decay=0.0,
        objective=nn.CrossEntropyLoss(), auprc=False, save='best.pt', validtime=False, objective_args_dict=None, input_to_float=True, clip_val=8,
        track_complexity=True, device=None, is_load=False, beta=1.0, tolerance=7, fold_index=-1, fuse_need=None, type="mixer"):
   
 
    print(f'task = {task}')
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


    model = SingleMDL(encoder, head=head, index=index, has_padding=is_packed, fuse_need=fuse_need).to(device)
    
    
    
    print(f'is_packed = {is_packed}')
    # if is_load:
    #     if os.path.exists(save):
    #         model = torch.load(save)
    #         print("Load saved model <Mixer>")

    def _trainprocess():
        
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False
            
        additional_params = []
        for m in additional_optimizing_modules:
            additional_params.extend(
                [p for p in m.parameters() if p.requires_grad])
        
            for p in m.parameters():
                print(p.requires_grad)    
            
        op = optimtype([p for p in model.parameters() if p.requires_grad] +
                       additional_params, lr=lr, weight_decay=weight_decay)
        bestvalloss = 10000
        bestacc = 0
        bestf1 = 0
        patience = 0

        def _processinput(inp):
            if input_to_float:
                return inp.float()
            else:
                return inp
            
        save_path = save + "head_" + type + "_single_view_" + str(index) + "_fold_" + str(fold_index) + ".pt"
        
        for epoch in range(total_epochs):
            totalloss = 0.0
            totals = 0
            model.train()
            for j in train_dataloader:
                op.zero_grad()
                if is_packed:
                    with torch.backends.cudnn.flags(enabled=False):
                        model.train()
                        out = model([[_processinput(i).to(device)
                                    for i in j[0]], j[1]])
                else:
                    model.train()
                    input = j[index]
                    out = model(_processinput(input).to(device))
                    # labels = j[-1]
                if not (objective_args_dict is None):
                    objective_args_dict['inputs'] = j[:-1]
                    objective_args_dict['training'] = True
                    objective_args_dict['model'] = model
                    objective_args_dict['device'] = device
                    objective_args_dict['reps'] = model.reps
                    objective_args_dict['fused'] = model.fuseout
                    loss = deal_with_objective(
                    objective, out, j[-1], objective_args_dict)
                else:
                    args = {}
                    args['device'] = device
                    loss = deal_with_objective(
                        objective, out, j[-1], args)
            

                totalloss += loss * len(j[-1])
                totals += len(j[-1])
                
                loss.backward()
                # print('>>>loss.backward()')
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                op.step()
            print("Epoch "+str(epoch)+" train loss: "+str((totalloss/totals).item()))
            validstarttime = time.time()
            if validtime:
                print("train total: "+str(totals))
            model.eval()
            with torch.no_grad():
                totalloss = 0.0
                pred = []
                true = []
                pts = []
                for j in valid_dataloader:
                    if is_packed:
                        out = model([[_processinput(i).to(device)
                                    for i in j[0]], j[1]])
                    else:
                        input = j[index]
                        out = model(_processinput(input).to(device))

                    if not (objective_args_dict is None):
                        objective_args_dict['reps'] = model.reps
                        objective_args_dict['fused'] = model.fuseout
                        objective_args_dict['inputs'] = j[:-1]
                        objective_args_dict['training'] = False
                        objective_args_dict['device'] = device
                        loss = deal_with_objective(
                        objective, out, j[-1], objective_args_dict)
                    else:
                        args = {}
                        args['device'] = device
                        loss = deal_with_objective(objective, out, j[-1], args)
                    totalloss += loss*len(j[-1])
                    
                    if task == "classification":
                        pred.append(torch.argmax(out, 1))
                    elif task == "multilabel":
                        pred.append(torch.sigmoid(out).round())
                    true.append(j[-1])
                    if auprc:
                        # pdb.set_trace()
                        sm = softmax(out)
                        pts += [(sm[i][1].item(), j[-1][i].item())
                                for i in range(j[-1].size(0))]
            if pred:
                pred = torch.cat(pred, 0)
            true = torch.cat(true, 0)
            totals = true.shape[0]
            valloss = totalloss/totals
            if task == "classification":
                acc = accuracy(true, pred)
                print("Epoch "+str(epoch)+" valid loss: "+str(valloss.item()) +
                      " acc: "+str(acc))
                if acc > bestacc:
                    patience = 0
                    bestacc = acc
                    print("Saving Best")
                    torch.save(model, save_path)
                else:
                    patience += 1
            elif task == "multilabel":
                f1_micro = f1_score(true, pred, average="micro")
                f1_macro = f1_score(true, pred, average="macro")
                print("Epoch "+str(epoch)+" valid loss: "+str(valloss) +
                      " f1_micro: "+str(f1_micro)+" f1_macro: "+str(f1_macro))
                if f1_macro > bestf1:
                    patience = 0
                    bestf1 = f1_macro
                    print("Saving Best")
                    torch.save(model, save)
                else:
                    patience += 1
            elif task == "regression":
                print("Epoch "+str(epoch)+" valid loss: "+str(valloss.item()))
                if valloss < bestvalloss:
                    patience = 0
                    bestvalloss = valloss
                    print("Saving Best")
                    torch.save(model, save_path)
                else:
                    patience += 1
            if early_stop and patience > tolerance:
                print(f'early_stop and patience > {tolerance}')
                break
            if auprc:
                print("AUPRC: "+str(AUPRC(pts)))
            validendtime = time.time()
            if validtime:
                print("valid time:  "+str(validendtime-validstarttime))
                print("Valid total: "+str(totals))
            
        if task == "classification":
            final_best = bestacc
        elif task == 'regression':
            final_best = bestvalloss
        elif task == 'multilabel':
            final_best = bestf1
        print(f'Final best = {final_best}')
        return final_best
        
    track_complexity = False
    if track_complexity:
        return all_in_one_train(_trainprocess, [model]+additional_optimizing_modules)
    else:
        return _trainprocess()
    
def train_head_single_modality_mfm(encoder, index, head, train_dataloader, valid_dataloader, total_epochs, additional_optimizing_modules=[], is_packed=False,
        early_stop=False, task="classification", optimtype=torch.optim.RMSprop, lr=0.001, weight_decay=0.0,
        objective=nn.CrossEntropyLoss(), auprc=False, save='best.pt', validtime=False, objective_args_dict=None, input_to_float=True, clip_val=8,
        track_complexity=True, device=None, is_load=False, beta=1.0, tolerance=7, fold_index=-1, fuse_need=None, type="mixer"):
   
 
    print(f'task = {task}')
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


    model = SingleMDL(encoder, head=head, index=index, has_padding=is_packed, fuse_need=fuse_need).to(device)
    
    
    
    print(f'is_packed = {is_packed}')
    # if is_load:
    #     if os.path.exists(save):
    #         model = torch.load(save)
    #         print("Load saved model <Mixer>")

    def _trainprocess():
     
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False

        head.eval()
        for p in head.parameters():
            p.requires_grad = False       
        
            
        additional_params = []
        
        for m in additional_optimizing_modules:
           
            for p in m.parameters():
               p.requires_grad = False
               
            additional_params.extend(
                [p for p in m.parameters() if p.requires_grad])
        
   
            
        op = optimtype([p for p in model.parameters() if p.requires_grad] +
                       additional_params, lr=lr, weight_decay=weight_decay)
        
        bestvalloss = 10000
        bestacc = 0
        bestf1 = 0
        patience = 0

        def _processinput(inp):
            if input_to_float:
                return inp.float()
            else:
                return inp
            
        save_path = save + "head_" + type + "_single_view_" + str(index) + "_fold_" + str(fold_index) + ".pt"
        
        for epoch in range(total_epochs):
            totalloss = 0.0
            totals = 0
            model.train()
            for j in train_dataloader:
                op.zero_grad()
                if is_packed:
                    with torch.backends.cudnn.flags(enabled=False):
                        model.train()
                        out = model([[_processinput(i).to(device)
                                    for i in j[0]], j[1]])
                else:
                    model.train()
                    input = j[index]
                    out = model(_processinput(input).to(device))
                    # labels = j[-1]
                if not (objective_args_dict is None):
                    objective_args_dict['inputs'] = j[:-1]
                    objective_args_dict['training'] = True
                    objective_args_dict['model'] = model
                    objective_args_dict['device'] = device
                    objective_args_dict['reps'] = model.reps
                    objective_args_dict['fused'] = model.fuseout
                    loss = deal_with_objective(
                    objective, out, j[-1], objective_args_dict)
                else:
                    args = {}
                    args['device'] = device
                    loss = deal_with_objective(
                        objective, out, j[-1], args)
            

                totalloss += loss * len(j[-1])
                totals += len(j[-1])
                
                loss.backward()
                # print('>>>loss.backward()')
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                op.step()
            print("Epoch "+str(epoch)+" train loss: "+str((totalloss/totals).item()))
            validstarttime = time.time()
            if validtime:
                print("train total: "+str(totals))
            model.eval()
            with torch.no_grad():
                totalloss = 0.0
                pred = []
                true = []
                pts = []
                for j in valid_dataloader:
                    if is_packed:
                        out = model([[_processinput(i).to(device)
                                    for i in j[0]], j[1]])
                    else:
                        input = j[index]
                        out = model(_processinput(input).to(device))

                    if not (objective_args_dict is None):
                        objective_args_dict['reps'] = model.reps
                        objective_args_dict['fused'] = model.fuseout
                        objective_args_dict['inputs'] = j[:-1]
                        objective_args_dict['training'] = False
                        objective_args_dict['device'] = device
                        loss = deal_with_objective(
                        objective, out, j[-1], objective_args_dict)
                    else:
                        args = {}
                        args['device'] = device
                        loss = deal_with_objective(objective, out, j[-1], args)
                    totalloss += loss*len(j[-1])
                    
                    if task == "classification":
                        pred.append(torch.argmax(out, 1))
                    elif task == "multilabel":
                        pred.append(torch.sigmoid(out).round())
                    true.append(j[-1])
                    if auprc:
                        # pdb.set_trace()
                        sm = softmax(out)
                        pts += [(sm[i][1].item(), j[-1][i].item())
                                for i in range(j[-1].size(0))]
            if pred:
                pred = torch.cat(pred, 0)
            true = torch.cat(true, 0)
            totals = true.shape[0]
            valloss = totalloss/totals
            if task == "classification":
                acc = accuracy(true, pred)
                print("Epoch "+str(epoch)+" valid loss: "+str(valloss.item()) +
                      " acc: "+str(acc))
                if acc > bestacc:
                    patience = 0
                    bestacc = acc
                    print("Saving Best")
                    torch.save(model, save_path)
                else:
                    patience += 1
            elif task == "multilabel":
                f1_micro = f1_score(true, pred, average="micro")
                f1_macro = f1_score(true, pred, average="macro")
                print("Epoch "+str(epoch)+" valid loss: "+str(valloss) +
                      " f1_micro: "+str(f1_micro)+" f1_macro: "+str(f1_macro))
                if f1_macro > bestf1:
                    patience = 0
                    bestf1 = f1_macro
                    print("Saving Best")
                    torch.save(model, save)
                else:
                    patience += 1
            elif task == "regression":
                print("Epoch "+str(epoch)+" valid loss: "+str(valloss.item()))
                if valloss < bestvalloss:
                    patience = 0
                    bestvalloss = valloss
                    print("Saving Best")
                    torch.save(model, save_path)
                else:
                    patience += 1
            if early_stop and patience > tolerance:
                print(f'early_stop and patience > {tolerance}')
                break
            if auprc:
                print("AUPRC: "+str(AUPRC(pts)))
            validendtime = time.time()
            if validtime:
                print("valid time:  "+str(validendtime-validstarttime))
                print("Valid total: "+str(totals))
            
        if task == "classification":
            final_best = bestacc
        elif task == 'regression':
            final_best = bestvalloss
        elif task == 'multilabel':
            final_best = bestf1
        print(f'Final best = {final_best}')
        return final_best
        
    track_complexity = False
    if track_complexity:
        return all_in_one_train(_trainprocess, [model]+additional_optimizing_modules)
    else:
        return _trainprocess()

def train_head_single_modality_mfm_train_head(encoder, index, head, train_dataloader, valid_dataloader, total_epochs, additional_optimizing_modules=[], is_packed=False,
        early_stop=False, task="classification", optimtype=torch.optim.RMSprop, lr=0.001, weight_decay=0.0,
        objective=nn.CrossEntropyLoss(), auprc=False, save='best.pt', validtime=False, objective_args_dict=None, input_to_float=True, clip_val=8,
        track_complexity=True, device=None, is_load=False, beta=1.0, tolerance=7, fold_index=-1, fuse_need=None, type="mixer"):

 
    print(f'task = {task}')
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


    model = SingleMDL(encoder, head=head, index=index, has_padding=is_packed, fuse_need=fuse_need).to(device)

    
    print(f'is_packed = {is_packed}')
    # if is_load:
    #     if os.path.exists(save):
    #         model = torch.load(save)
    #         print("Load saved model <Mixer>")

    def _trainprocess():

        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False


        # head.eval()
        # for p in head.parameters():
        #     p.requires_grad = False       
        
            
        additional_params = []
        
        for m in additional_optimizing_modules:
   
            for p in m.parameters():
               p.requires_grad = False
               
            additional_params.extend(
                [p for p in m.parameters() if p.requires_grad])
        
   
            
        op = optimtype([p for p in model.parameters() if p.requires_grad] +
                       additional_params, lr=lr, weight_decay=weight_decay)
        
        bestvalloss = 10000
        bestacc = 0
        bestf1 = 0
        patience = 0

        def _processinput(inp):
            if input_to_float:
                return inp.float()
            else:
                return inp
            
        save_path = save + "head_" + type + "_single_view_" + str(index) + "_fold_" + str(fold_index) + ".pt"
        
        for epoch in range(total_epochs):
            totalloss = 0.0
            totals = 0
            model.train()
            for j in train_dataloader:
                op.zero_grad()
                if is_packed:
                    with torch.backends.cudnn.flags(enabled=False):
                        model.train()
                        out = model([[_processinput(i).to(device)
                                    for i in j[0]], j[1]])
                else:
                    model.train()
                    input = j[index]
                    out = model(_processinput(input).to(device))
                    # labels = j[-1]
                if not (objective_args_dict is None):
                    objective_args_dict['inputs'] = j[:-1]
                    objective_args_dict['training'] = True
                    objective_args_dict['model'] = model
                    objective_args_dict['device'] = device
                    objective_args_dict['reps'] = model.reps
                    objective_args_dict['fused'] = model.fuseout
                    loss = deal_with_objective(
                    objective, out, j[-1], objective_args_dict)
                else:
                    args = {}
                    args['device'] = device
                    loss = deal_with_objective(
                        objective, out, j[-1], args)
            

                totalloss += loss * len(j[-1])
                totals += len(j[-1])
                
                loss.backward()
                # print('>>>loss.backward()')
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                op.step()
            print("Epoch "+str(epoch)+" train loss: "+str((totalloss/totals).item()))
            validstarttime = time.time()
            if validtime:
                print("train total: "+str(totals))
            model.eval()
            with torch.no_grad():
                totalloss = 0.0
                pred = []
                true = []
                pts = []
                for j in valid_dataloader:
                    if is_packed:
                        out = model([[_processinput(i).to(device)
                                    for i in j[0]], j[1]])
                    else:
                        input = j[index]
                        out = model(_processinput(input).to(device))

                    if not (objective_args_dict is None):
                        objective_args_dict['reps'] = model.reps
                        objective_args_dict['fused'] = model.fuseout
                        objective_args_dict['inputs'] = j[:-1]
                        objective_args_dict['training'] = False
                        objective_args_dict['device'] = device
                        loss = deal_with_objective(
                        objective, out, j[-1], objective_args_dict)
                    else:
                        args = {}
                        args['device'] = device
                        loss = deal_with_objective(objective, out, j[-1], args)
                    totalloss += loss*len(j[-1])
                    
                    if task == "classification":
                        pred.append(torch.argmax(out, 1))
                    elif task == "multilabel":
                        pred.append(torch.sigmoid(out).round())
                    true.append(j[-1])
                    if auprc:
                        # pdb.set_trace()
                        sm = softmax(out)
                        pts += [(sm[i][1].item(), j[-1][i].item())
                                for i in range(j[-1].size(0))]
            if pred:
                pred = torch.cat(pred, 0)
            true = torch.cat(true, 0)
            totals = true.shape[0]
            valloss = totalloss/totals
            if task == "classification":
                acc = accuracy(true, pred)
                print("Epoch "+str(epoch)+" valid loss: "+str(valloss.item()) +
                      " acc: "+str(acc))
                if acc > bestacc:
                    patience = 0
                    bestacc = acc
                    print("Saving Best")
                    torch.save(model, save_path)
                else:
                    patience += 1
            elif task == "multilabel":
                f1_micro = f1_score(true, pred, average="micro")
                f1_macro = f1_score(true, pred, average="macro")
                print("Epoch "+str(epoch)+" valid loss: "+str(valloss) +
                      " f1_micro: "+str(f1_micro)+" f1_macro: "+str(f1_macro))
                if f1_macro > bestf1:
                    patience = 0
                    bestf1 = f1_macro
                    print("Saving Best")
                    torch.save(model, save)
                else:
                    patience += 1
            elif task == "regression":
                print("Epoch "+str(epoch)+" valid loss: "+str(valloss.item()))
                if valloss < bestvalloss:
                    patience = 0
                    bestvalloss = valloss
                    print("Saving Best")
                    torch.save(model, save_path)
                else:
                    patience += 1
            if early_stop and patience > tolerance:
                print(f'early_stop and patience > {tolerance}')
                break
            if auprc:
                print("AUPRC: "+str(AUPRC(pts)))
            validendtime = time.time()
            if validtime:
                print("valid time:  "+str(validendtime-validstarttime))
                print("Valid total: "+str(totals))
            
        if task == "classification":
            final_best = bestacc
        elif task == 'regression':
            final_best = bestvalloss
        elif task == 'multilabel':
            final_best = bestf1
        print(f'Final best = {final_best}')
        return final_best
        
    track_complexity = False
    if track_complexity:
        return all_in_one_train(_trainprocess, [model]+additional_optimizing_modules)
    else:
        return _trainprocess()

    
def valid_single_modality(model, index,  valid_dataloader, is_packed=False, input_to_float=True,
     task="classification", objective=nn.CrossEntropyLoss(),  objective_args_dict=None, 
    device=None, beta=1.0, fold_index=-1, fuse_need=None):
    
    def _processinput(inp):
        if input_to_float:
            return inp.float()
        else:
            return inp
        
    model.eval()
    
    with torch.no_grad():
        
        totalloss = 0.0
        pred = []
        true = []
        
        
        for j in valid_dataloader:
            if is_packed:
                out = model([[_processinput(i).to(device)
                            for i in j[0]], j[1]])
            else:
                input = j[index]
                out = model(_processinput(input).to(device))

            if not (objective_args_dict is None):
                objective_args_dict['reps'] = model.reps
                objective_args_dict['fused'] = model.fuseout
                objective_args_dict['inputs'] = j[:-1]
                objective_args_dict['training'] = False
                objective_args_dict['device'] = device
                loss = deal_with_objective(
                        objective, out, j[-1], objective_args_dict)
            else:
                args = {}
                args['device'] = device
                loss = deal_with_objective(objective, out, j[-1], args)
            totalloss += loss*len(j[-1])
                    
            if task == "classification":
                pred.append(torch.argmax(out, 1))
            elif task == "multilabel":
                pred.append(torch.sigmoid(out).round())
            true.append(j[-1])

    if pred:
        pred = torch.cat(pred, 0)
    
    true = torch.cat(true, 0)
            
    totals = true.shape[0]
    valloss = totalloss/totals
            
    if task == "classification":
        acc = accuracy(true, pred)
        print("Fold "+str(fold_index)+" valid loss: "+str(valloss.item()) +
                      " acc: "+str(acc))
        final_best = acc
        
    elif task == "multilabel":
        f1_micro = f1_score(true, pred, average="micro")
        f1_macro = f1_score(true, pred, average="macro")
        print("Fold "+str(fold_index)+" valid loss: "+str(valloss) +
                      " f1_micro: "+str(f1_micro)+" f1_macro: "+str(f1_macro))
        final_best = f1_macro

    elif task == "regression":
        print("Fold "+str(fold_index)+" valid loss: "+str(valloss.item()))
        final_best = valloss

    print(f'Final best = {final_best}')
    return final_best


def valid_fused_model(model, index,  valid_dataloader, additional_optimizing_modules=[], is_packed=False, task="classification", objective=nn.CrossEntropyLoss(),  objective_args_dict=None, input_to_float=True, device=None, fold_index=-1):
    
    def _processinput(inp):
        if input_to_float:
            return inp.float()
        else:
            return inp    
        
    model.eval()
    with torch.no_grad():
        totalloss = 0.0
        pred = []
        true = []

        for j in valid_dataloader:
            if is_packed:
                out = model([[_processinput(j[0][index]).to(device)
                                    for i in j[0]], j[1]])
            else:
                out = model([_processinput(j[index]).to(device)
                                    for i in j[:-1]])

            if not (objective_args_dict is None):
                objective_args_dict['reps'] = model.reps
                objective_args_dict['fused'] = model.fuseout
                objective_args_dict['inputs'] = j[:-1]
                objective_args_dict['training'] = False
                objective_args_dict['device'] = device
                loss = deal_with_objective(
                objective, out, j[-1], objective_args_dict)
            else:
                args = {}
                args['device'] = device
                loss = deal_with_objective(objective, out, j[-1], args)
            totalloss += loss*len(j[-1])
                    
            if task == "classification":
                pred.append(torch.argmax(out, 1))
            elif task == "multilabel":
                pred.append(torch.sigmoid(out).round())
            true.append(j[-1])

    if pred:
        pred = torch.cat(pred, 0)
    true = torch.cat(true, 0)
    totals = true.shape[0]
    valloss = totalloss/totals
    if task == "classification":
        acc = accuracy(true, pred)
        print("Fold "+str(fold_index)+" valid loss: "+str(valloss.item()) +
                      " acc: "+str(acc))
        final_best = acc
        
    elif task == "multilabel":
        f1_micro = f1_score(true, pred, average="micro")
        f1_macro = f1_score(true, pred, average="macro")
        
        print("Fold "+str(fold_index)+" valid loss: "+str(valloss) +
                      " f1_micro: "+str(f1_micro)+" f1_macro: "+str(f1_macro))
        
        final_best = f1_macro

    elif task == "regression":
        print("Fold "+str(fold_index)+" valid loss: "+str(valloss.item()))

        final_best = valloss

    return final_best
        





def single_test(
        model, test_dataloader, is_packed=False,
        criterion=nn.CrossEntropyLoss(), task="classification", auprc=False, input_to_float=True, device=None):
    """Run single test for model.

    Args:
        model (nn.Module): Model to test
        test_dataloader (torch.utils.data.Dataloader): Test dataloader
        is_packed (bool, optional): Whether the input data is packed or not. Defaults to False.
        criterion (_type_, optional): Loss function. Defaults to nn.CrossEntropyLoss().
        task (str, optional): Task to evaluate. Choose between "classification", "multiclass", "regression", "posneg-classification". Defaults to "classification".
        auprc (bool, optional): Whether to get AUPRC scores or not. Defaults to False.
        input_to_float (bool, optional): Whether to convert inputs to float before processing. Defaults to True.
    """
    def _processinput(inp):
        if input_to_float:
            return inp.float()
        else:
            return inp
    with torch.no_grad():
        if device is None:
            device_str = "cuda:0"
            device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        totalloss = 0.0
        pred = []
        true = []
        pts = []
        for j in test_dataloader:
            model.eval()
            if is_packed:
                out = model([[_processinput(i).to(device)
                            for i in j[0]], j[1]])
            else:
                out = model([_processinput(i).float().to(device)
                            for i in j[:-1]])
            if type(criterion) == torch.nn.modules.loss.BCEWithLogitsLoss or type(criterion) == torch.nn.MSELoss:
                loss = criterion(out, j[-1].float().to(device))

            # elif type(criterion) == torch.nn.CrossEntropyLoss:
            #     loss=criterion(out, j[-1].long().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))

            elif type(criterion) == nn.CrossEntropyLoss:
                if len(j[-1].size()) == len(out.size()):
                    truth1 = j[-1].squeeze(len(out.size())-1)
                else:
                    truth1 = j[-1]
                loss = criterion(out, truth1.long().to(device))
            else:
                loss = criterion(out, j[-1].to(device))
            totalloss += loss*len(j[-1])
            if task == "classification":
                pred.append(torch.argmax(out, 1))
            elif task == "multilabel":
                pred.append(torch.sigmoid(out).round())
            elif task == "posneg-classification":
                prede = []
                oute = out.cpu().numpy().tolist()
                for i in oute:
                    if i[0] > 0:
                        prede.append(1)
                    elif i[0] < 0:
                        prede.append(-1)
                    else:
                        prede.append(0)
                pred.append(torch.LongTensor(prede))
            true.append(j[-1])
            if auprc:
                # pdb.set_trace()
                sm = softmax(out)
                pts += [(sm[i][1].item(), j[-1][i].item())
                        for i in range(j[-1].size(0))]
        if pred:
            pred = torch.cat(pred, 0)
        true = torch.cat(true, 0)
        totals = true.shape[0]
        testloss = totalloss/totals
        if auprc:
            print("AUPRC: "+str(AUPRC(pts)))
        if task == "classification":
            print("acc: "+str(accuracy(true, pred)))
            return {'Accuracy': accuracy(true, pred)}
        elif task == "multilabel":
            print(" f1_micro: "+str(f1_score(true, pred, average="micro")) +
                  " f1_macro: "+str(f1_score(true, pred, average="macro")))
            return {'micro': f1_score(true, pred, average="micro"), 'macro': f1_score(true, pred, average="macro")}
        elif task == "regression":
            print("mse: "+str(testloss.item()))
            return {'MSE': testloss.item()}
        elif task == "posneg-classification":
            trueposneg = true
            accs = eval_affect(trueposneg, pred)
            acc2 = eval_affect(trueposneg, pred, exclude_zero=False)
            print("acc: "+str(accs) + ', ' + str(acc2))
            return {'Accuracy': accs}



def test(
        model, test_dataloaders_all, dataset='default', method_name='My method', is_packed=False, criterion=nn.CrossEntropyLoss(), task="classification", auprc=False, input_to_float=True, no_robust=False, mixer_type=None, fold_index=-1):
    """
    Handle getting test results for a simple supervised training loop.
    
    :param model: saved checkpoint filename from train
    :param test_dataloaders_all: test data
    :param dataset: the name of dataset, need to be set for testing effective robustness
    :param criterion: only needed for regression, put MSELoss there   
    """
    if no_robust:
        def _testprocess():
            single_test(model, test_dataloaders_all, is_packed,
                        criterion, task, auprc, input_to_float)
        all_in_one_test(_testprocess, [model])
        return

    def _testprocess():
        single_test(model, test_dataloaders_all[list(test_dataloaders_all.keys())[
                    0]][0], is_packed, criterion, task, auprc, input_to_float)
    all_in_one_test(_testprocess, [model])
    for noisy_modality, test_dataloaders in test_dataloaders_all.items():
        print("Testing on noisy data ({})...".format(noisy_modality))
        robustness_curve = dict()
        for test_dataloader in tqdm(test_dataloaders):
            single_test_result = single_test(
                model, test_dataloader, is_packed, criterion, task, auprc, input_to_float)
            for k, v in single_test_result.items():
                curve = robustness_curve.get(k, [])
                curve.append(v)
                robustness_curve[k] = curve
        for measure, robustness_result in robustness_curve.items():
            robustness_key = '{} {}'.format(dataset, noisy_modality)
            print("relative robustness ({}, {}): {}".format(noisy_modality, measure, str(
                relative_robustness(robustness_result, robustness_key))))
            if len(robustness_curve) != 1:
                robustness_key = '{} {}'.format(robustness_key, measure)
            print("effective robustness ({}, {}): {}".format(noisy_modality, measure, str(
                effective_robustness(robustness_result, robustness_key))))
            if fold_index == -1:
                if mixer_type is None:
                    fig_name = '{}-{}-{}-{}'.format(method_name,
                                            robustness_key, noisy_modality, measure)
                else:
                    fig_name = '{}-{}-{}-{}-{}'.format(method_name,
                                            robustness_key, noisy_modality, measure, mixer_type)
            elif  mixer_type is None:
                
                fig_name = '{}-{}-{}-{}-{}'.format(method_name,
                                            robustness_key, noisy_modality, measure, fold_index)
            else:
                fig_name = '{}-{}-{}-{}-{}-{}'.format(method_name,
                                            robustness_key, noisy_modality, measure, fold_index, mixer_type)
                
                
            single_plot(robustness_result, robustness_key, xlabel='Noise level',
                        ylabel=measure, fig_name=fig_name, method=method_name)
            print("Plot saved as "+fig_name)



def mixer_transport_visualization(model, test_dataloader, dataset='default', method_name='My method', is_packed=False, criterion=nn.CrossEntropyLoss(), task="classification", auprc=False, input_to_float=True, no_robust=False, device=None, prefix='', fold_index='', plots_dir=''):
    def _processinput(inp):
        if input_to_float:
            return inp.float()
        else:
            return inp
        
    with torch.no_grad():
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        for j in test_dataloader:
            model.eval()
            if auprc:
                pass
            if task == "classification":
                pass
            elif task == "multilabel":
                pass
            elif task == "posneg-classification":
                pass
            elif task == "regression":
                
              
                if len(j[-1].shape) > 2:
                    flat_coords = j[-1].reshape(j[-1].shape[0], -1)
                    dis_origin = distance_tensor(flat_coords.to(device), flat_coords.to(device))
                else:
                    dis_origin = distance_tensor(j[-1].to(device), j[-1].to(device))
                average_dis = torch.mean(dis_origin)
                median_dis = torch.median(dis_origin)
                print(f'>> Truth: Average = {average_dis}, Median = {median_dis} of distance matrix.')
                
                title =  ' Modality i <--> 0, origin truth distance'
                if torch.cuda.is_available():
                    plt.imshow(dis_origin.cpu().numpy(), cmap='viridis', interpolation='nearest')
                else:
                    plt.imshow(dis_origin.numpy(), cmap='viridis', interpolation='nearest')
                plt.title(title)
                save_name = prefix  + '_origin_truth_from_Modality_i_to_0.png'
                plt.savefig(plots_dir + save_name)
                # print(f'Plot {save_name}') 
                
                
                if is_packed:
                    zs = model.encode([[_processinput(i).to(device)  for i in j[0]], j[1]])  
                else:
                    zs = model.encode([_processinput(i).to(device) for i in j[:-1]])

                trans = mixer_fuse(zs=zs.copy())
                
                trans_average = []
                trans_median = []
                
                for i in range(0, len(zs)-1):

                    nonzero_indices = trans[i].nonzero()

                    corresponding_values = dis_origin[nonzero_indices[:, 0], nonzero_indices[:, 1]]
                    
                    average_i = torch.mean(corresponding_values)
                    median_i = torch.median(corresponding_values)
                    
                    print(f'>> Modality {i+1} : Average = {average_i}, Median = {median_i} transport map to z0')
                    trans_average.append(average_i)
                    trans_median.append(median_i)
     
                    plt.clf()
                    if torch.cuda.is_available():
                        plt.imshow(trans[i].cpu().numpy(), cmap='viridis', interpolation='nearest')
                    else:
                        plt.imshow(trans[i].numpy(), cmap='viridis', interpolation='nearest')
                    plt.title('Fold '+ str(fold_index) + ':  Modality ' + str(i+1) + ' <--> 0, transport map')
                    save_name = prefix  + '_transfold' + str(fold_index) + '_Modality' + str(i+1)+ '.pdf'
                    plt.savefig(plots_dir + save_name)

                plt.colorbar() 
            
                trans_average_merge = torch.stack(trans_average)
                trans_median_merge = torch.stack(trans_median)
                
                trans_average_total_average = torch.mean(trans_average_merge)
                trans_median_total_average = torch.mean(trans_median_merge)
                
                print(f'>> Total Average: Mean = {trans_average_total_average},  Median = {trans_median_total_average} for all modality transport maps. ')
            break
        
        # plt.close()