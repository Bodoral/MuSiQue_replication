import json
import pandas as pd
from typing import Dict, List


def to_jsonl(data: List[Dict], file_path: str):
    with open(file_path, mode="w", encoding="utf-8") as file:
        for line in data:
            json.dump(line, file, ensure_ascii=False)
            file.write("\n")


def find_init_two_hop_questions(
    heads_df: pd.DataFrame, tails_df: pd.DataFrame = None, switch: bool = False
) -> pd.DataFrame:

    if tails_df is None:
        tails_df = heads_df
    if switch:
        holder = heads_df
        heads_df = tails_df
        tails_df = holder

    renaming_tails_df = {
        col: col.replace("head", "mid") for col in tails_df.columns if "head" in col
    }
    tail_suffix = (
        "_" + list(renaming_tails_df.items())[0][1].split("_")[-1]
        if renaming_tails_df
        else ""
    )
    new_tail_suffix = "_tail" if not renaming_tails_df else ""

    renaming_heads_df = {
        col: col.replace("tail", "mid") for col in heads_df.columns if "tail" in col
    }
    head_suffix = "_mid" if renaming_heads_df else ""
    new_head_suffix = "_head"

    return heads_df.rename(renaming_heads_df, axis=1)[
        ~heads_df["answer_entity" + head_suffix]
        .isna()
        ].merge(
            tails_df.rename(renaming_tails_df, axis=1),
            how="inner",
            left_on="answer_entity" + head_suffix,
            right_on="question_entity" + tail_suffix,
            suffixes=(new_head_suffix, new_tail_suffix),
    )


def find_multi_hop_questions(
    heads_df: pd.DataFrame,
    tails_df: pd.DataFrame = None,
    switch: bool = False,
    hops: int = 3,
) -> pd.DataFrame:

    if tails_df is None:
        tails_df = heads_df
    if switch:
        holder = heads_df
        heads_df = tails_df
        tails_df = holder

    renaming_tails_df = {
        col: col.replace("head", "mid") for col in tails_df.columns if "head" in col
    }
    tail_suffix = (
        "_" + list(renaming_tails_df.items())[0][1].split("_")[-1]
        if renaming_tails_df
        else ""
    )

    renaming_heads_df = {
        col: col.replace("tail", "mid") for col in heads_df.columns if "tail" in col
    }
    head_suffix = "_mid" if renaming_heads_df else ""

    return (
        heads_df.rename(renaming_heads_df, axis=1)
        .merge(
            tails_df.rename(renaming_tails_df, axis=1),
            how="inner",
            left_on="question" + head_suffix,
            right_on="question" + tail_suffix,
            suffixes=(None, hops - 1),
        )
        .rename(
            {
                f"question_entity_mid{hops-1}": f"question_entity{hops-1}_mid",
                f"answer_entity_mid{hops-1}": f"answer_entity{hops-1}_mid",
                f"passage_mid{hops-1}": f"passage{hops-1}_mid",
            },
            axis=1,
        )
    )


def find_adjacent_head(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[~df["question_tail"].isna()]
        .merge(
            df,
            how="inner",
            left_on="question_tail",
            right_on="question_tail",
            suffixes=("1", "2"),
        )
        .rename(
            {
                "answer_entity_tail1": "answer_entity_tail",
                "answer_entity_tail2": "answer_entity2_tail",
                "question_entity_tail1": "question_entity_tail",
                "question_entity_tail2": "question_entity2_tail",
                "passage_tail1": "passage_tail",
                "passage_tail2": "passage2_tail",
            },
            axis=1,
        )
    )


def filter_questions_where_head_and_tail_form_cycle(
    df: pd.DataFrame, head_suffix: str = "_head", tail_suffix: str = "_tail"
) -> pd.DataFrame:
    return df[
        (df["question" + head_suffix] != df["question" + tail_suffix])
        & (df["answer_entity" + tail_suffix] != df["question_entity" + head_suffix])
        & (df["passage" + head_suffix] != df["passage" + tail_suffix])
    ]


def filter_questions_where_head_and_tail_form_cycle_loop_v(
    df: pd.DataFrame, head_suffix: str = "_head", tail_suffix: str = "_tail"
) -> pd.DataFrame:
    filtered_df = df[df["question" + head_suffix] != df["question" + tail_suffix]]

    heads_entity_col_suffixes = set(
        [
            col.split("question_entity")[-1]
            for col in df.columns
            if col.startswith("question_entity") and col.endswith(head_suffix)
        ]
    )
    for head_suffix_ in heads_entity_col_suffixes:
        filtered_df = filtered_df[
            (
                filtered_df["answer_entity" + tail_suffix]
                != filtered_df["question_entity" + head_suffix_]
            )
            & (
                filtered_df["passage" + head_suffix_]
                != filtered_df["passage" + tail_suffix]
            )
        ]
    return filtered_df


def filter_identical_heads(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["question_head1"] != df["question_head2"])
        & (df["question_entity_head1"] != df["question_entity_head2"])
        & (df["passage_head1"] != df["passage_head2"])
    ]


def rename_mid_node(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    return df.rename(
        {
            "question_mid": f"question{suffix}",
            "question_entity_mid": f"question_entity{suffix}",
            "answer_entity_mid": f"answer_entity{suffix}",
            "passage_mid": f"passage{suffix}",
            "question_entity2_mid": f"question_entity2{suffix}",
            "answer_entity2_mid": f"answer_entity2{suffix}",
            "passage2_mid": f"passage2{suffix}",
        },
        axis=1,
    )


def restore_questions_info(df: pd.DataFrame, data: List[Dict]) -> List[Dict]:
    question_ids_cols = [
        col for col in df.columns if col.startswith("question") and "entity" not in col
    ]
    df = df[question_ids_cols].drop_duplicates()
    df_questions_with_info = pd.DataFrame()
    for col in question_ids_cols:
        df_questions_with_info[col] = df[col].map(lambda x: data[x])

    return df_questions_with_info.to_dict(orient="records")


def save_debugging_info(df_dict: Dict):
    for k, v in df_dict.items():
        to_jsonl(v.to_dict(orient="records"), str(k) + ".jsonl")


def composable_questions(
    records_of_entities: List[Dict], qa_data: List[Dict], debug: bool = True
) -> Dict:
    """
    find sets of composable questions from a large pool of single-hop questions.
    these sets can be used to compose multi-hop questions.
    The returned dictionary contains various type of sets:
    - 2hop    0--->0
    - 2hop_with_adjacent_head     0--->0<---0
    - 3hop    0--->0--->0
    - 3hop_with_adjacent_head     0--->0--->0<---0
    - 3hop_with_adjacent_head2
        0--->
                0--->0
        0--->
    - 4hop    0--->0--->0--->0

    Args:
    - records_of_entities: list of dictionaries for each per of question and answer entities [{"question":x,"question_entity": e1.id ,"answer_entity":e2.id, "passage":123},{"question":x,"question_entity": e3.id ,"answer_entity":e2.id, "passage":123}..]
    - qa_data: list of dictionaries, each contains a question, answer and passage
    - debug: if True will save additional info that is helpful for debugging
    """
    df = pd.DataFrame(records_of_entities)
    # 0--->0
    df_2hop = df.pipe(find_init_two_hop_questions).pipe(
        filter_questions_where_head_and_tail_form_cycle
    )
    # 0--->0<---0
    df_2hop_with_adjacent_head = df_2hop.pipe(find_adjacent_head).pipe(
        filter_identical_heads
    )
    # 0--->0--->0
    df_3hop = df_2hop.pipe(find_multi_hop_questions).pipe(
        filter_questions_where_head_and_tail_form_cycle
    )
    # 0--->0--->0<---0
    df_3hop_with_adjacent_head = (
        df_2hop_with_adjacent_head.pipe(
            find_multi_hop_questions, tails_df=df_2hop, hops=4
        )
        .pipe(filter_questions_where_head_and_tail_form_cycle, head_suffix="_head1")
        .pipe(filter_questions_where_head_and_tail_form_cycle, head_suffix="_head2")
    )
    # 0--->
    #       0--->0
    # 0--->
    df_3hop_with_adjacent_head2 = df_2hop_with_adjacent_head.pipe(
        find_multi_hop_questions, tails_df=df_2hop, hops=4, switch=True
    ).pipe(filter_questions_where_head_and_tail_form_cycle)
    # 0--->0--->0--->0
    df_4hop = (
        df_3hop.pipe(rename_mid_node, suffix="_mid0")
        .pipe(find_multi_hop_questions, tails_df=df_2hop, hops=4)
        .pipe(filter_questions_where_head_and_tail_form_cycle)
        .pipe(
            filter_questions_where_head_and_tail_form_cycle_loop_v, head_suffix="_mid0"
        )
    )

    if debug:
        save_debugging_info(
            {
                "2hop": df_2hop,
                "2hop_with_adjacent_head": df_2hop_with_adjacent_head,
                "3hop": df_3hop,
                "3hop_with_adjacent_head": df_3hop_with_adjacent_head,
                "3hop_with_adjacent_head2": df_3hop_with_adjacent_head2,
                "4hop": df_4hop,
            }
        )
        to_jsonl(qa_data, "debug_info.jsonl")

    return {
        "2hop": restore_questions_info(df_2hop, qa_data),
        "2hop_with_adjacent_head": restore_questions_info(df_2hop_with_adjacent_head, qa_data),
        "3hop": restore_questions_info(df_3hop, qa_data),
        "3hop_with_adjacent_head": restore_questions_info(df_3hop_with_adjacent_head, qa_data),
        "3hop_with_adjacent_head2": restore_questions_info(df_3hop_with_adjacent_head2, qa_data),
        "4hop": restore_questions_info(df_4hop, qa_data),
    }
