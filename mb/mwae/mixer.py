import torch
from collections.abc import Iterable
from mwae.get_distance import mixer_fuse
import copy


def is_same_type(obj1, obj2):
    if isinstance(obj1, torch.Tensor):
        if isinstance(obj1, torch.Tensor):
            return True
        else:
            return False
    elif isinstance(obj1, Iterable):
        if isinstance(obj2, Iterable):
            if len(obj1) == len(obj2):
                for i in range(len(obj1)):
                    return is_same_type(obj1[i], obj2[i])
            else:
                return False
        else:
            return False
    elif type(obj1) == type(obj2):
        return True
    else:
        return False
        

def mixer_data(ms):
    num_views = len(ms)
    trans = mixer_fuse(zs=ms.copy())
    nms=[]
    for j in range(num_views):
   
        if j==0:
            # x = torch.transpose(trans[j], 0, 1) 
            # x = torch.eye(trans[j].shape[0]).to(trans[j].device)
            nms.append(ms[j])
            continue
        
        x = trans[j-1]
        
        y = ms[j]
        
        
        if len(ms[j].shape) > 2:
            y = y.reshape(y.shape[0], -1)
        
     
        value = torch.matmul(x, y)
     
        if len(ms[j].shape) > 2:
            value = value.reshape_as(ms[j])

        nms.append(value)
        
    return nms


def mixer_aug2(zs, labels):

    unique_labels, indices = torch.unique(labels, return_inverse=True)
    num_views = len(zs)
    zs_aug = copy.copy(zs)
    labels_aug = torch.zeros_like(labels)
    start = 0

  
    type_divide=[[0]]
    for j in range(num_views):
        if j > 0:
            not_added = True
            for tt in range(len(type_divide)):
                type_emu = type_divide[tt]
                if is_same_type(zs[type_emu[0]], zs[j]):
                    type_emu.append(j)
                    not_added = False
                    break
            if not_added:
                type_divide.append([j])   

    for label in unique_labels:
        
        mask = (labels == label).squeeze()
        true_count = torch.sum(mask).item()             
       
        for type_emu in type_divide:
            if len(type_emu) == 1:
             
                modality_index = type_emu[0]
                modality_data = zs[modality_index][mask]
                zs_aug[modality_index][start:start+true_count] = modality_data
                print(f'>>> Augmented modality {modality_index}.')
            else:
                ms = [] 
                
                te0 = zs[type_emu[0]]
                
                if isinstance(te0, torch.Tensor):
                   
                    for modality_index in type_emu:
                        tmp = zs[modality_index][mask].squeeze()
                        ms.append(tmp)
                    
                    ms_mix_aug = mixer_data(ms)
                    relative_index = 0
                  
                    for modality_index in type_emu:
                        zss1 = zs_aug[modality_index][start:start+true_count]
                        zss2 = ms_mix_aug[relative_index]
                        if zss1.shape == zss2.shape:
                            zs_aug[modality_index][start:start+true_count] = zss2
                        else:
                            zs_aug[modality_index][start:start+true_count] = torch.unsqueeze(zss2, dim=-1)
                        relative_index += 1
                        print(f'>>> Augmented modality {modality_index}.')
                
                elif isinstance(te0, Iterable):
                    len_iter = len(te0)
                    for il in range(len_iter):
                        ms=[]
                        item = te0[il]
                        if isinstance(item, torch.Tensor):
                       
                            for modality_index in type_emu:
                                tmp = zs[modality_index][il][mask].squeeze()
                                ms.append(tmp)
                        
                            ms_mix_aug = mixer_data(ms)
                            relative_index = 0
                          
                            for modality_index in type_emu:
                                zss1 = zs_aug[modality_index][il][start:start+true_count]
                                zss2 = ms_mix_aug[relative_index]
                                if  zss1.shape == zss2.shape:
                                    zs_aug[modality_index][il][start:start+true_count] = zss2
                                else:
                                    zs_aug[modality_index][il][start:start+true_count] = torch.unsqueeze(zss2, dim=-1)
                                relative_index += 1
                                print(f'>>> Augmented modality {modality_index} for index={il}.')
                        elif isinstance(item, Iterable):
                            len_iter = len(item)
                            ms=[]
                            for iil in range(len_iter):
                                item_iil = item[iil]
                                if isinstance(item_iil, torch.Tensor):
                                  
                                    for modality_index in type_emu:
                                        tmp = zs[modality_index][il][iil][mask]
                                        ms.append(tmp)
                                  
                                    ms_mix_aug = mixer_data(ms.copy())
                                    relative_index = 0
                                   
                                    for modality_index in type_emu:
                                        zss1 = zs_aug[modality_index][il][iil][start:start+true_count]
                                        zss2 = ms_mix_aug[relative_index]
                                        if zss1.shape == zss2.shape:
                                            zs_aug[modality_index][il][iil][start:start+true_count] = zss2
                                            print(f'>>> Augmented modality {modality_index} for index={il} with index_iil={iil}.')
                                        else:
                                           
                                            print("Unsure how to unsqueeze ?...")
                                        relative_index += 1
       
        labels_aug[start:start+true_count]=label
        start = start+true_count
    return zs_aug, labels_aug



def mixer_aug(zs, labels):

    unique_labels, indices = torch.unique(labels, return_inverse=True)
    num_views = len(zs)
    zs_aug = copy.copy(zs)
    labels_aug = torch.zeros_like(labels)
    start = 0

   
    type_divide=[[0]]
    for j in range(num_views):
        if j > 0:
            not_added = True
            for tt in range(len(type_divide)):
                type_emu = type_divide[tt]
                if is_same_type(zs[type_emu[0]], zs[j]):
                    type_emu.append(j)
                    not_added = False
                    break
            if not_added:
                type_divide.append([j])   

    for label in unique_labels:
        mask = (labels == label).squeeze()
        true_count = torch.sum(mask).item()             
     
        for type_emu in type_divide:
            if len(type_emu) == 1:
               
                modality_index = type_emu[0]
                modality_data = zs[modality_index][mask]
                zs_aug[modality_index][start:start+true_count] = modality_data
                # print(f'>>>  Label: {label} Not Augmented modality {modality_index}.')
            else:
                ms = [] 
               
                te0 = zs[type_emu[0]]
              
                if isinstance(te0, torch.Tensor):
                   
                    for modality_index in type_emu:
                        tmp = zs[modality_index][mask]
                        ms.append(tmp)
                    
                    ms_mix_aug = mixer_data(ms.copy())
                    relative_index = 0
                  
                    for modality_index in type_emu:
                        zss1 = zs_aug[modality_index][start:start+true_count]
                        zss2 = ms_mix_aug[relative_index]
                        if zss1.shape == zss2.shape:
                            zs_aug[modality_index][start:start+true_count] = zss2
                            # print(f'>>>  Label: {label} Augmented modality {modality_index}.')
                        else:
                            modality_data = zs[modality_index][mask]
                            zs_aug[modality_index][start:start+true_count] = modality_data
                            # print(f'>>> Label: {label}  Unsure modality {modality_index}???')
                            # zs_aug[modality_index][start:start+true_count] = torch.unsqueeze(zss2, dim=-1)
                        relative_index += 1
                        
                elif isinstance(te0, Iterable):
                    len_iter = len(te0)
                    for il in range(len_iter):
                        ms=[]
                        item = te0[il]
                        if isinstance(item, torch.Tensor):
                          
                            for modality_index in type_emu:
                                tmp = zs[modality_index][il][mask]
                                ms.append(tmp)
                           
                            ms_mix_aug = mixer_data(ms.copy())
                            relative_index = 0
                        
                            for modality_index in type_emu:
                                zss1 = zs_aug[modality_index][il][start:start+true_count]
                                zss2 = ms_mix_aug[relative_index]
                                if  zss1.shape == zss2.shape:
                                    zs_aug[modality_index][il][start:start+true_count] = zss2
                                    # print(f'>>> Label: {label} Augmented modality {modality_index} for index={il}.')
                                else:
                                    modality_data = zs[modality_index][mask]
                                    zs_aug[modality_index][start:start+true_count] = modality_data
                                    # print(f'>>> Label: {label} Unsure modality {modality_index} for index={il} ???')
                                relative_index += 1
                                
                        elif isinstance(item, Iterable):
                            len_iter = len(item)
                            for iil in range(len_iter):
                                item_iil = item[iil]
                                if isinstance(item_iil, torch.Tensor):
                                
                                    ms = []
                                    for modality_index in type_emu:
                                        tmp = zs[modality_index][il][iil][mask]
                                        ms.append(tmp)
                                
                                    ms_mix_aug = mixer_data(ms.copy())
                                    relative_index = 0
                                 
                                    for modality_index in type_emu:
                                        zss1 = zs_aug[modality_index][il][iil][start:start+true_count]
                                        zss2 = ms_mix_aug[relative_index]
                                        if zss1.shape == zss2.shape:
                                            zs_aug[modality_index][il][iil][start:start+true_count] = zss2
                                            # print(f'>>> Label {label}: Augmented modality {modality_index} for index={il} with index_iil={iil}.')
                                        else:
                                          
                                            modality_data = zs[modality_index][il][iil][mask]
                                            zs_aug[modality_index][il][iil][start:start+true_count] = modality_data
                                        relative_index += 1
     
        labels_aug[start:start+true_count]=label
        start = start+true_count
    return zs_aug, labels_aug


def mixer_aug_regression(zs, truth):
    
    # torch.autograd.set_detect_anomaly(True)
    
    num_views = len(zs)
    trans = mixer_fuse(zs=zs.copy())
    aug_zs = []
    aug_truth = []
    tmp_p = truth.clone()
    
    for j in range(num_views):
     
        if j==0:
            aug_zs.append(zs[j])
            continue
        
       
        tmp_z = zs[j].clone()
        
      
        zj = zs[j]
        truthj = tmp_p
        
        if len(tmp_z.shape) > 2:
            zj = zj.reshape(tmp_z.shape[0], -1)
        if len(tmp_p.shape) > 2:
            truthj= tmp_p.reshape(tmp_z.shape[0], -1)
        
       
        zj = torch.matmul(trans[j-1], zj)
        truthj = torch.matmul(trans[j-1], truthj)
        
   
        if len(tmp_z.shape) > 2:
            zj = zj.reshape_as(tmp_z)
        if len(tmp_p.shape) > 2:
            truthj= truthj.reshape_as(tmp_p)  
        
       
        aug_zs.append(zj)
        aug_truth.append(truthj) 

   
    final_truth = tmp_p
    for item in aug_truth:
        final_truth += item
    final_truth /= num_views
    
    return aug_zs, final_truth

