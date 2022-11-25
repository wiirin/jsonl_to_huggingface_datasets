import json
from datasets import Dataset, ClassLabel, Sequence


class jsonl_to_hfdatasets:
    def __init__(self, jsonl_data):
        self.data = jsonl_data
        self.labels_to_ids, self.ids_to_labels = self.__get_unique_labels()

    def __get_unique_labels(self):
        label_unique = set()
        for file in self.data:
            for label in file["label"]:
                label_unique.add(label[2])
        label_unique = list(label_unique)
        iob_labels = []
        for l in label_unique:
            iob_labels.append("B-" + l)
            iob_labels.append("I-" + l)
        iob_labels.append("0")

        labels_to_ids = {k: v for v, k in enumerate(sorted(iob_labels))}
        ids_to_labels = {v: k for v, k in enumerate(sorted(iob_labels))}
        return labels_to_ids, ids_to_labels

    def convert_to_hf_dataset(self):
        tokens = []
        ner_labels = []
        ids = []

        for file in self.data:
            ids.append(file["id"])
            word = ""
            token = []
            ner_label = []
            labels = {}

            for label in file["label"]:
                labels[label[0]] = (label[1], label[2])

            end_postition = -1
            for i, char in enumerate(file["text"]):
                if i in labels.keys():
                    annotated_words = file["text"][i : labels[i][0]].split(" ")
                    if len(annotated_words) > 1:
                        token.extend(annotated_words)
                        ner_label.append("B-" + labels[i][1])
                        ner_label.extend(
                            ["I-" + labels[i][1]] * (len(annotated_words) - 1)
                        )
                    else:
                        token.append(file["text"][i : labels[i][0]])
                        ner_label.append("B-" + labels[i][1])
                    end_postition = labels[i][0]
                elif (char == " ") & (len(word) > 0):
                    token.append(word)
                    ner_label.append("0")
                    word = ""
                elif (char == " ") & (len(word) == 0):
                    pass
                elif i > end_postition:
                    word += char

            ner_tags = []
            for ner in ner_label:
                ner_tags.append(self.labels_to_ids[ner])

            tokens.append(token)
            ner_labels.append(ner_label)

        data = {"id": ids, "tokens": tokens, "ner_tags": ner_labels}
        ds = Dataset.from_dict(data)
        ds.features["ner_tags"] = Sequence(ClassLabel(names=self.labels_to_ids.keys()))
        return ds


if __name__ == "__main__":
    with open("annotation_data.jsonl", "r") as file:
        data = file.readlines()
    jsonl_data = [json.loads(d) for d in data]
    convert = jsonl_to_hfdatasets(jsonl_data)
    result = convert.convert_to_hf_dataset()
