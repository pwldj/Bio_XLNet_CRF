import json
import conlleval


def get_result(eval_file, out, decode):
    with open(eval_file, 'r') as f:
        lines = json.load(f)

    all_logit = []
    all_label = []
    for l in lines:
        logit = []
        label = []
        for i, m in enumerate(l['label_mask']):
            if m:
                logit.append(l['logits'][i])
                label.append(l["labels"][i])
        all_logit.extend(logit[:-3])
        all_label.extend(label[:-3])

    assert len(all_logit) == len(all_label)

    evalseq = []
    for i in range(len(all_logit)):
        evalseq.append("{} {} {}".format(i, decode[int(all_label[i])] if int(all_label[i]) in decode.keys() else "O",
                                         decode[int(all_logit[i])] if int(all_logit[i]) in decode.keys() else "O",))

    count = conlleval.evaluate(evalseq)
    conlleval.report(count, out)
