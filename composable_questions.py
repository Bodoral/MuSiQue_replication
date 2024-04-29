import argparse
import json
import pandas as pd
import tqdm
from entity_recognition import NER


def find_composable_questions(heads_df, tails_df = None, switch = False):
    if tails_df is None:
        tails_df = heads_df
    if switch:
        holder = heads_df
        heads_df = tails_df
        tails_df = holder

    renaming_tails_df = { col:col.replace("head","mid") for col in tails_df.columns if "head" in col}
    tail_suffix = "_"+ list(renaming_tails_df.items())[0][1].split("_")[-1] if renaming_tails_df else ""
    new_tail_suffix = "_tail" if not renaming_tails_df else ""

    renaming_heads_df = { col:col.replace("tail","mid") for col in heads_df.columns if "tail" in col}
    head_suffix = "_mid" if renaming_heads_df else ""
    new_head_suffix = "_head"

    return (heads_df.rename(renaming_heads_df,axis = 1)
            .merge(
                tails_df.rename(renaming_tails_df,axis = 1), 
                how = "inner", 
                left_on="answer_entity"+head_suffix,
                right_on="question_entity"+tail_suffix,
                suffixes=(new_head_suffix,new_tail_suffix)
                )
            )  

def find_multi_hop_questions(heads_df, tails_df = None, switch = False, hops = 3):
    if tails_df is None:
        tails_df = heads_df
    if switch:
        holder = heads_df
        heads_df = tails_df
        tails_df = holder

    renaming_tails_df = { col:col.replace("head","mid") for col in tails_df.columns if "head" in col}
    tail_suffix = "_"+ list(renaming_tails_df.items())[0][1].split("_")[-1] if renaming_tails_df else ""
    
    renaming_heads_df = { col:col.replace("tail","mid") for col in heads_df.columns if "tail" in col}
    head_suffix = "_mid" if renaming_heads_df else ""

    return (heads_df.rename(renaming_heads_df,axis = 1)
            .merge(
                tails_df.rename(renaming_tails_df,axis = 1), 
                how = "inner", 
                left_on="question"+head_suffix,
                right_on="question"+tail_suffix,
                suffixes=(None,hops-1)
                )
            .rename({ f"question_entity_mid{hops-1}": f"question_entity{hops-1}_mid",
                    f"answer_entity_mid{hops-1}":f"answer_entity{hops-1}_mid",
                    f"passage_mid{hops-1}": f"passage{hops-1}_mid",
                    },axis = 1
                    )                    
            )

def find_adjencet_head(df):
    return (df
            .merge(
                df, 
                how = "inner", 
                left_on = "question_tail", 
                right_on="question_tail", 
                suffixes=("1","2")
                )
            .rename({"answer_entity_tail1":"answer_entity_tail", 
                            "answer_entity_tail2": "answer_entity2_tail",
                            "question_entity_tail1": "question_entity_tail",
                            "question_entity_tail2":"question_entity2_tail",
                            "passage_tail1":"passage_tail",
                             "passage_tail2":"passage2_tail"
                            },axis = 1
                    )
    )

def filter_questions_where_head_and_tail_form_cycle(df, head_suffix = "_head",tail_suffix = "_tail"):
    return df[
        (df["question"+head_suffix] != df["question"+tail_suffix])
        &
        (df["answer_entity"+tail_suffix] != df["question_entity"+head_suffix])
        &
        (df["passage"+head_suffix] != df["passage"+tail_suffix])
        ]

def filter_questions_where_head_and_tail_form_cycle_loop_v(df, head_suffix = "_head",tail_suffix = "_tail"):
    filtered_df = df[df["question"+head_suffix] != df["question"+tail_suffix]]

    heads_entity_col_suffixes = set([col.split("question_entity")[-1] for col in df.columns if col.startswith("question_entity") and col.endswith(head_suffix)])
    for head_suffix_ in heads_entity_col_suffixes:
        filtered_df = filtered_df[
            (filtered_df["answer_entity"+tail_suffix] != filtered_df["question_entity"+head_suffix_])
            &
            (filtered_df["passage"+head_suffix_] != filtered_df["passage"+tail_suffix])
            ]
    return filtered_df
    
def filter_identical_heads(df):
    return df[
        (df["question_head1"] != df["question_head2"])
        &
        (df["question_entity_head1"] != df["question_entity_head2"])
        &
        (df["passage_head1"] != df["passage_head2"])
        ]

def rename_mid_node(df, suffix):
    return df.rename(
        {'question_mid':f'question{suffix}' , 
            'question_entity_mid':f'question_entity{suffix}', 
            'answer_entity_mid': f'answer_entity{suffix}',
            'passage_mid':f'passage{suffix}',
        'question_entity2_mid':f'question_entity2{suffix}',
        'answer_entity2_mid':f'answer_entity2{suffix}',
        'passage2_mid':f'passage2{suffix}'
        },
        axis = 1
    )

def restore_questions_info(df, data:list[dict]):
    question_ids_cols =[ col for col in df.columns if col.startswith("question") and "entity" not in col ]
    df_questions_with_info = pd.DataFrame()
    for col in question_ids_cols:
        df_questions_with_info[col] = df[col].map(lambda x: data[x])
    
    return df_questions_with_info.to_dict(orient = "records")

def save_debugging_info(df_dict):
    for k, v in df_dict.items():
        write_jsonl_file(v.to_dict(orient = "records"), str(k)+".jsonl")

def composable_questions(records_of_entities:list[dict], qa_data:list[dict], debug = False):
    df = pd.DataFrame(records_of_entities)
    # 0--->0
    df_2hop = df.pipe(find_composable_questions).pipe(filter_questions_where_head_and_tail_form_cycle)
    # 0--->0<---0
    df_2hop_with_adjacent_head = df_2hop.pipe(find_adjencet_head).pipe(filter_identical_heads)
    # 0--->0--->0
    df_3hop = df_2hop.pipe(find_multi_hop_questions).pipe(filter_questions_where_head_and_tail_form_cycle)
    # 0--->0--->0<---0
    df_3hop_with_adjacent_head = (df_2hop_with_adjacent_head
                                    .pipe(find_multi_hop_questions, tails_df = df_2hop, hops = 4)
                                    .pipe(filter_questions_where_head_and_tail_form_cycle, head_suffix = "_head1")
                                    .pipe(filter_questions_where_head_and_tail_form_cycle, head_suffix = "_head2")
                                    )
    # 0--->
    #       0--->0
    # 0--->
    df_3hop_with_adjacent_head2 = (df_2hop_with_adjacent_head
                                    .pipe(find_multi_hop_questions, tails_df = df_2hop, hops = 4, switch = True)
                                    .pipe(filter_questions_where_head_and_tail_form_cycle)
                                    )
    # 0--->0--->0--->0
    df_4hop = (df_3hop
                .pipe(rename_mid_node, suffix = "_mid0")
                .pipe(find_multi_hop_questions,tails_df =df_2hop, hops = 4)
                .pipe(filter_questions_where_head_and_tail_form_cycle)
                .pipe(filter_questions_where_head_and_tail_form_cycle_loop_v, head_suffix = "_mid0")
                )
    
    if debug:
        save_debugging_info(
            {
                "2hop_questions":df_2hop,
                "2hop_questions_with_adjacent_head":df_2hop_with_adjacent_head,
                "3hop_questions":df_3hop,
                "3hop_questions_with_adjacent_head":df_3hop_with_adjacent_head,
                "3hop_questions_with_adjacent_head2":df_3hop_with_adjacent_head2,
                "4hop_questions":df_4hop,
                }
        )


    return {
        "2hop_questions":restore_questions_info(df_2hop, qa_data),
        "2hop_questions_with_adjacent_head":restore_questions_info(df_2hop_with_adjacent_head, qa_data),
        "3hop_questions":restore_questions_info(df_3hop, qa_data),
        "3hop_questions_with_adjacent_head":restore_questions_info(df_3hop_with_adjacent_head,qa_data),
        "3hop_questions_with_adjacent_head2":restore_questions_info(df_3hop_with_adjacent_head2,qa_data),
        "4hop_questions":restore_questions_info(df_4hop,qa_data),

    }

def single_file_worker(file_name,args,entity_recognizer):
    cpu_batch_size =args.cpu_batch_size
    gpu_batch_size = args.gpu_batch_size
    out_files = args.out_files
    output_file_prefix = ".".join(file_name.split("/")[-1].split(".")[:-1])
    data = read_jsonl_file(file_name)

    # 1.recognize entities in questions and answers
    print("#task 1: recognize and link entities in questions and answers")
    records = []
    for i in tqdm.tqdm(range(0, len(data), gpu_batch_size)):
        questions_metadata = entity_recognizer.recognize_and_link_multi_docs([q["question"] for q in data[i:i+gpu_batch_size]])
        answers_metadata = entity_recognizer.recognize_and_link_multi_docs( [" , ".join([ ans for ans in q["answers"]]) for q in data[i:i+gpu_batch_size]]) # multi answers
        for j, question_metadata, answer_metadata in zip(range(i,i+gpu_batch_size), questions_metadata, answers_metadata):
            records.extend(
                [
                    {"question":j,"question_entity": q_entity["id"],"answer_entity":ans_entity["id"], "passage":data[j]["passages"][0]["id"]}
                    for q_entity in question_metadata  for ans_entity in  answer_metadata
                ]
                )
            

    # 2.find composable questions sets
    print("#Task 3: composable questions")
    composable_questions_sets = composable_questions(records_of_entities= records,qa_data = data, debug= True)
    print("saving output files...")
    for sets in composable_questions_sets.keys():
         write_jsonl_file(composable_questions_sets[sets], f"{out_files}/{output_file_prefix}_{sets}_composable_questions.jsonl")


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description= "TODO")
    parser.add_argument("-inf","--in_files",required=True, help=", delimited list of jsonl files", type=lambda s: [str(item) for item in s.split(",")])
    parser.add_argument("-outf","--out_files", help="where to save output files, default is working directory ", type=str,default = "")
    parser.add_argument("-cpub","--cpu_batch_size", help="TODO", type= int, default= 64)
    parser.add_argument("-gpub","--gpu_batch_size", help="TODO", type= int, default=8 )

    args = parser.parse_args()
    files = args.in_files
    cpu_batch_size =args.cpu_batch_size
    gpu_batch_size = args.gpu_batch_size
    entity_recognizer = GenericNamedEntityLinking(no_cuda = False)

    for file in files:
        print(f"*****  Working on file: {file}  *****")
        single_file_worker(file,args,entity_recognizer)


        
                
    
