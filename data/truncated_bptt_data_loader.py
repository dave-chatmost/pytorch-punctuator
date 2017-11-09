import numpy as np


def get_one_batch_data(dataset, feats_utt, targets_utt, new_utt_flags,
                       batch_size, steps, idx):
    new_utt_flags = [0] * batch_size
    for b in range(batch_size):
        if feats_utt[b].shape[0] == 0:
            # print(idx)
            # TODO: make dataset an iterator?
            if idx == len(dataset): return None, None, None, idx, True
            feats, targets = dataset[idx]
            idx += 1
            if feats is not None:
                feats_utt[b] = feats
                targets_utt[b] = targets
                new_utt_flags[b] = 1
    
    # end the training after processing all the frames
    frames_to_go = 0
    for b in range(batch_size):
        frames_to_go += feats_utt[b].shape[0]
    if frames_to_go == 0: return None, None, None, idx, True

    #### START pack the mini-batch data ####
    feat_host = np.zeros((steps, batch_size))
    target_host = np.zeros((steps, batch_size))
    frame_num_utt = [0] * batch_size

    # slice at most 'batch_size' frames
    for b in range(batch_size):
        num_rows = feats_utt[b].shape[0]
        frame_num_utt[b] = min(steps, num_rows)

    # pack the features
    for b in range(batch_size):
        for t in range(frame_num_utt[b]):
            feat_host[t, b] = feats_utt[b][t]

    # pack the targets
    for b in range(batch_size):
        for t in range(frame_num_utt[b]):
            target_host[t, b] = targets_utt[b][t]
    #### END pack data ####

    # remove the data we just packed
    for b in range(batch_size):
        packed_rows = frame_num_utt[b]
        feats_utt[b] = feats_utt[b][packed_rows:]
        targets_utt[b] = targets_utt[b][packed_rows:]
        left_rows = feats_utt[b].shape[0]
        if left_rows < steps:
            feats_utt[b] = np.array([])
        # feats
        # rows = feats_utt[b].shape[0]
        # if rows == frame_num_utt[b]:
        #     feats_utt[b] = np.array([])
        # else:
        #     packed_rows = frame_num_utt[b]
        #     feats_utt[b] = feats_utt[b][packed_rows:]
        #     targets_utt[b] = targets_utt[b][packed_rows:]
    #### END prepare mini-batch data ####
    return feat_host, target_host, new_utt_flags, idx, False
