import argparse
import json
import tqdm
from entity_linking.nel import NEL
from composable_questions import composable_questions
from typing import Dict, List


def from_jsonl(file_path: str) -> List[Dict]:
    data = []
    with open(file_path, mode="r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))
    return data


def to_jsonl(data: List[Dict], file_path: str):
    with open(file_path, mode="w", encoding="utf-8") as file:
        for line in data:
            json.dump(line, file, ensure_ascii=False)
            file.write("\n")


def single_file_worker(in_file_path, args, entity_linker):
    """
    a worker that perform two tasks:
    1- NEL of questions and answers (this step include both entities recognition and disambiguation)
    2- finding sets of composable questions that are connected by an entity
    """

    gpu_batch_size = args.gpu_batch_size
    out_path = args.out_path
    output_file_prefix = ".".join(in_file_path.split("/")[-1].split(".")[:-1])
    data = from_jsonl(in_file_path)

    # 1.recognize entities in questions and answers
    print("#task 1: recognize and link entities in questions and answers")
    records = []
    for i in tqdm.tqdm(range(0, len(data), gpu_batch_size)):
        questions_metadata = entity_linker.link_entities_in_docs(
            [q["question"] for q in data[i : i + gpu_batch_size]]
        )
        answers_metadata = entity_linker.link_entities_in_docs(
            [
                " , ".join([ans for ans in q["answers"]])
                for q in data[i : i + gpu_batch_size]
            ]
        )  # multi answers
        for j, question_metadata, answer_metadata in zip(
            range(i, i + gpu_batch_size), questions_metadata, answers_metadata
        ):
            records.extend(
                [
                    {
                        "question": j,
                        "question_entity": q_entity.id,
                        "answer_entity": ans_entity.id,
                        "passage": data[j]["passage_id"],
                    }
                    for q_entity in question_metadata
                    for ans_entity in answer_metadata
                ]
            )

    # 2.find composable questions
    print("#task 2: find composable questions")
    composable_questions_sets = composable_questions(
        records_of_entities=records, qa_data=data, debug=False
    )

    print("saving output files...")
    for sets in composable_questions_sets.keys():
        to_jsonl(
            composable_questions_sets[sets],
            f"{out_path}/{output_file_prefix}_{sets}_composable_questions.jsonl",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-in",
        "--in_files",
        required=True,
        help=", delimited list of jsonl files",
        type=lambda s: [str(item) for item in s.split(",")],
    )
    parser.add_argument(
        "-out",
        "--out_path",
        help="where to save output files, default is working directory ",
        type=str,
        default=".",
    )
    parser.add_argument(
        "-cpub",
        "--cpu_batch_size",
        help="batch size for CPU tasks",
        type=int,
        default=64,
    )
    parser.add_argument(
        "-gpub",
        "--gpu_batch_size",
        help="batch size for GPU tasks",
        type=int,
        default=8,
    )

    args = parser.parse_args()
    files = args.in_files
    entity_linker = NEL(no_cuda=False)

    for file in files:
        print(f"*****  working on file: {file}  *****")
        single_file_worker(file, args, entity_linker)
