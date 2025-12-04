
def generate_data_keys_sequential(config, recs_list, verbose=True):
    import numpy as np
    from tqdm import tqdm
    from classes.annotation import Annotation

    PRE_EVENT_WINDOW = 30.0  # seconds
    segments = []

    for idx, f in tqdm(enumerate(recs_list), disable=not verbose):
        annotations = Annotation.loadAnnotation(config.data_path, f)
        rec_dur = float(annotations.rec_duration)

        # Create sequential segments across the whole recording
        n_segs = int(np.floor((np.floor(rec_dur) - config.frame) / config.stride))
        if n_segs <= 0:
            continue

        seg_start = np.arange(0, n_segs) * config.stride
        seg_stop = seg_start + config.frame

        # Initialise all labels to 0 (non-pre-ictal)
        labels = np.zeros(n_segs, dtype=int)

        # If there are events, mark 30-second pre-ictal windows as label 1
        if annotations.events:
            for ev in annotations.events:
                event_start = float(ev[0])  # onset in seconds

                # 30-second window before the event, clipped at 0
                pre_start = max(0.0, event_start - PRE_EVENT_WINDOW)
                pre_end = event_start

                # Mark segments whose START lies in [pre_start, pre_end)
                in_window = (seg_start >= pre_start) & (seg_start < pre_end)
                labels[in_window] = 1

                # NOTE: ictal segments (seg_start >= event_start) are intentionally
                # kept at label 0 so the model focuses on pre-ictal prediction only.

        # Store segments for this recording
        seg_keys = np.column_stack(([idx] * n_segs, seg_start, seg_stop, labels))
        segments.extend(seg_keys)

    return segments


# Helper used by all key generators to enforce the global binary pre-ictal labelling

def _label_preictal_binary(seg_start, events, pre_event_window=30.0):

    import numpy as np

    labels = np.zeros_like(seg_start, dtype=int)
    if not events:
        return labels

    for ev in events:
        event_start = float(ev[0])
        pre_start = max(0.0, event_start - pre_event_window)
        pre_end = event_start
        mask = (seg_start >= pre_start) & (seg_start < pre_end)
        labels[mask] = 1

    return labels


def generate_data_keys_sequential_window(config, recs_list, t_add, verbose=True):
    import numpy as np
    from tqdm import tqdm
    from classes.annotation import Annotation

    PRE_EVENT_WINDOW = 30.0
    segments = []

    for idx, f in tqdm(enumerate(recs_list), disable=not verbose):
        annotations = Annotation.loadAnnotation(config.data_path, f)

        if annotations.rec_duration < 600:
            print('short file: ' + f[0] + ' ' + f[1])

        # Collect all segments for this recording, then relabel in one pass
        local_segments = []

        if annotations.events:
            if len(annotations.events) == 1:
                ev = annotations.events[0]

                if t_add * 2 < ev[1] - ev[0]:
                    print('check batches!!!')
                    to_add_ev = 30
                else:
                    to_add_ev = t_add - np.round((ev[1] - ev[0]) / 2)

                to_add_plus = to_add_ev
                to_add_minus = to_add_ev

                if ev[1] + to_add_ev > np.floor(annotations.rec_duration) - config.frame:
                    to_add_plus = np.floor(annotations.rec_duration) - ev[1] - config.frame
                    to_add_minus = to_add_ev + to_add_ev - to_add_plus

                if ev[0] - to_add_ev < 0:
                    to_add_minus = ev[0] - 1
                    to_add_plus = to_add_ev + to_add_ev - to_add_minus

                if to_add_plus + to_add_minus + ev[1] - ev[0] > t_add * 2:
                    to_add_plus = to_add_plus - (to_add_plus + to_add_minus + ev[1] - ev[0] - t_add * 2)
                elif to_add_plus + to_add_minus + ev[1] - ev[0] < t_add * 2:
                    if to_add_plus == np.floor(annotations.rec_duration) - ev[1] - config.frame:
                        to_add_minus += (t_add * 2 - (to_add_plus + to_add_minus + ev[1] - ev[0]))
                    elif to_add_minus == ev[0] - 1:
                        to_add_plus += (t_add * 2 - (to_add_plus + to_add_minus + ev[1] - ev[0]))
                    else:
                        to_add_plus += (t_add * 2 - (to_add_plus + to_add_minus + ev[1] - ev[0]))

                if to_add_plus + to_add_minus + ev[1] - ev[0] != t_add * 2:
                    print('bad segmentation!!!')

                segs_nr = 0

                # pre-ictal and pre-event background
                n_segs = int(np.floor((ev[0] - (ev[0] - to_add_minus)) / config.stride) - 1)
                seg_start = np.arange(0, n_segs) * config.stride + ev[0] - to_add_minus
                seg_stop = seg_start + config.frame
                local_segments.extend(np.column_stack(([idx] * n_segs, seg_start, seg_stop, np.zeros(n_segs))))
                segs_nr += n_segs

                # ictal region (kept as 0 for now, will be relabelled globally)
                n_segs = int(np.floor((ev[1] - ev[0]) / config.stride) + 1)
                seg_start = np.arange(0, n_segs) * config.stride + ev[0] - config.stride
                seg_stop = seg_start + config.frame
                local_segments.extend(np.column_stack(([idx] * n_segs, seg_start, seg_stop, np.zeros(n_segs))))
                segs_nr += n_segs

                # post-ictal / background
                n_segs = int(np.floor(np.floor(ev[1] + to_add_plus - ev[1]) / config.stride))
                seg_start = np.arange(0, n_segs) * config.stride + ev[1]
                seg_stop = seg_start + config.frame
                local_segments.extend(np.column_stack(([idx] * n_segs, seg_start, seg_stop, np.zeros(n_segs))))
                segs_nr += n_segs

                if segs_nr != 600:
                    print('wrong nr segs')
            else:
                end_rec = False
                end_seg = 0
                for i, ev in enumerate(annotations.events):
                    skip = False
                    if t_add * 2 < ev[1] - ev[0]:
                        print('check batches!!!')
                        to_add_ev = 30
                    else:
                        to_add_ev = t_add - np.round((ev[1] - ev[0]) / 2)

                    if i == 0:
                        to_add_plus = to_add_ev
                        to_add_minus = to_add_ev

                        if ev[0] - to_add_ev < 0:
                            to_add_minus = ev[0] - 1
                            to_add_plus = to_add_ev + (to_add_ev - ev[0]) + 1

                        end_seg = to_add_plus

                    else:
                        if ev[0] > end_seg:
                            if ev[0] - to_add_ev > end_seg:
                                to_add_minus = to_add_ev
                                to_add_plus = to_add_ev
                            else:
                                to_add_minus = ev[0] - end_seg
                                to_add_plus = 2 * to_add_ev - to_add_minus
                        else:
                            if ev[1] > end_seg:
                                print('check boundary case')
                            else:
                                skip = True

                        end_seg = ev[1] + to_add_plus

                    if end_seg > np.floor(annotations.rec_duration) - config.frame - t_add * 2:
                        end_rec = True

                    if not skip and not end_rec:
                        if to_add_plus + to_add_minus + ev[1] - ev[0] > t_add * 2:
                            to_add_plus -= (to_add_plus + to_add_minus + ev[1] - ev[0] - t_add * 2)
                        elif to_add_plus + to_add_minus + ev[1] - ev[0] < t_add * 2:
                            if to_add_plus == annotations.rec_duration - ev[1] - config.frame:
                                to_add_minus += (t_add * 2 - (to_add_plus + to_add_minus + ev[1] - ev[0]))
                            elif to_add_minus == ev[0] - 1:
                                to_add_plus += (t_add * 2 - (to_add_plus + to_add_minus + ev[1] - ev[0]))
                            else:
                                to_add_plus += (t_add * 2 - (to_add_plus + to_add_minus + ev[1] - ev[0]))

                        if to_add_plus + to_add_minus + ev[1] - ev[0] != t_add * 2:
                            print('bad segmentation!!!')

                        if ev[1] + to_add_plus >= np.floor(annotations.rec_duration) - config.frame:
                            to_add_plus = np.floor(annotations.rec_duration) - config.frame - ev[1]
                            to_add_minus = to_add_ev + (to_add_ev - to_add_plus)

                        segs_nr = 0

                        n_segs = int(np.floor((ev[0] - (ev[0] - to_add_minus)) / config.stride) - 1)
                        if n_segs < 0:
                            n_segs = 0
                        seg_start = np.arange(0, n_segs) * config.stride + ev[0] - to_add_minus
                        seg_stop = seg_start + config.frame
                        local_segments.extend(np.column_stack(([idx] * n_segs, seg_start, seg_stop, np.zeros(n_segs))))
                        segs_nr += n_segs

                        n_segs = int(np.floor((ev[1] - ev[0]) / config.stride) + 1)
                        seg_start = np.arange(0, n_segs) * config.stride + ev[0] - config.stride
                        seg_stop = seg_start + config.frame
                        local_segments.extend(np.column_stack(([idx] * n_segs, seg_start, seg_stop, np.zeros(n_segs))))
                        segs_nr += n_segs

                        n_segs = int(np.floor(np.floor(ev[1] + to_add_plus - ev[1]) / config.stride))
                        if n_segs < 0:
                            n_segs = 0
                        seg_start = np.arange(0, n_segs) * config.stride + ev[1]
                        seg_stop = seg_start + config.frame
                        local_segments.extend(np.column_stack(([idx] * n_segs, seg_start, seg_stop, np.zeros(n_segs))))
                        segs_nr += n_segs

                    elif skip and not end_rec:
                        # This branch in the original code only relabelled seizure segments;
                        # here we just note the overlap; labels will be recomputed globally.
                        n_segs = int(np.floor((ev[1] - ev[0]) / config.stride) + 1)
                        seg_start = np.arange(0, n_segs) * config.stride + ev[0] - config.stride
                        # No direct extension here; overlapping segments are already in local_segments.

                    if segs_nr != 600:
                        print('wrong nr segs')

        if local_segments:
            local_segments = np.vstack(local_segments)
            seg_start_all = local_segments[:, 1]
            new_labels = _label_preictal_binary(seg_start_all, annotations.events, PRE_EVENT_WINDOW)
            local_segments[:, 3] = new_labels
            segments.extend(local_segments)

    return segments


def generate_data_keys_subsample(config, recs_list, verbose=True):
    import numpy as np
    import random
    from tqdm import tqdm
    from classes.annotation import Annotation

    PRE_EVENT_WINDOW = 30.0

    # label 1
    positive_segments = []  
    # label 0
    negative_segments = []  

    for idx, f in tqdm(enumerate(recs_list), disable=not verbose):
        annotations = Annotation.loadAnnotation(config.data_path, f)

        # Collect all segments for this recording, using the original
        # segmentation scheme (seizure and non-seizure windows), but we
        # will relabel everything afterwards.
        local_segments = []

        if not annotations.events:
            # No events: just sequential non-seizure segments
            n_segs = int(np.floor((np.floor(annotations.rec_duration) - config.frame) / config.stride))
            seg_start = np.arange(0, n_segs) * config.stride
            seg_stop = seg_start + config.frame
            local_segments.extend(np.column_stack(([idx] * n_segs, seg_start, seg_stop, np.zeros(n_segs))))
        else:
            for e, ev in enumerate(annotations.events):
                # High-resolution segments around the event (original 'seizure' region)
                n_segs = int(((ev[1] + config.frame * (1 - config.boundary)) -
                              (ev[0] - config.frame * (1 - config.boundary)) -
                              config.frame) / config.stride_s)
                seg_start = np.arange(0, n_segs) * config.stride_s + ev[0] - config.frame * (1 - config.boundary)
                seg_stop = seg_start + config.frame
                local_segments.extend(np.column_stack(([idx] * n_segs, seg_start, seg_stop, np.zeros(n_segs))))

                # Background between events with coarser stride
                if e == 0:
                    n_segs = int(np.floor((ev[0]) / config.stride) - 1)
                    seg_start = np.arange(0, n_segs) * config.stride
                    seg_stop = seg_start + config.frame
                    if n_segs < 0:
                        n_segs = 0
                    local_segments.extend(np.column_stack(([idx] * n_segs, seg_start, seg_stop, np.zeros(n_segs))))
                else:
                    n_segs = int(np.floor((ev[0] - annotations.events[e - 1][1]) / config.stride) - 1)
                    if n_segs < 0:
                        n_segs = 0
                    seg_start = np.arange(0, n_segs) * config.stride + annotations.events[e - 1][1]
                    seg_stop = seg_start + config.frame
                    local_segments.extend(np.column_stack(([idx] * n_segs, seg_start, seg_stop, np.zeros(n_segs))))

                if e == len(annotations.events) - 1:
                    n_segs = int(np.floor((np.floor(annotations.rec_duration) - ev[1]) / config.stride) - 1)
                    seg_start = np.arange(0, n_segs) * config.stride + ev[1]
                    seg_stop = seg_start + config.frame
                    local_segments.extend(np.column_stack(([idx] * n_segs, seg_start, seg_stop, np.zeros(n_segs))))

        if not local_segments:
            continue

        local_segments = np.vstack(local_segments)
        seg_start_all = local_segments[:, 1]
        labels = _label_preictal_binary(seg_start_all, annotations.events, PRE_EVENT_WINDOW)
        local_segments[:, 3] = labels

        positive_segments.extend(local_segments[labels == 1])
        negative_segments.extend(local_segments[labels == 0])

    if len(positive_segments) == 0:
        print('No pre-ictal segments found; returning only negative segments')
        segments = negative_segments
    else:
        # Subsample negatives
        n_pos = len(positive_segments)
        n_neg_desired = config.factor * n_pos
        if len(negative_segments) > n_neg_desired:
            negative_sample = random.sample(negative_segments, n_neg_desired)
        else:
            negative_sample = negative_segments

        segments = list(positive_segments) + list(negative_sample)
        random.shuffle(segments)

    return segments
