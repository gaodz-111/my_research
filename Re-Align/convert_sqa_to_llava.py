import json
import os
import fire
import re





def get_question_text(problem):
    question = problem['question']
    return question


def get_context_text(problem, use_caption):
    txt_context = problem['hint']
    img_context = problem['caption'] if use_caption else ""
    context = " ".join([txt_context, img_context]).strip()
    if context == "":
        context = "N/A"
    return context


def get_choice_text(probelm, options):
    choices = probelm['choices']
    choice_list = []
    for i, c in enumerate(choices):
        choice_list.append("({}) {}".format(options[i], c))
    choice_txt = " ".join(choice_list)
    #print(choice_txt)
    return choice_txt


def get_answer(problem, options):
    return options[problem['answer']]


def get_lecture_text(problem):
    # \\n: GPT-3 can generate the lecture with more tokens.
    lecture = problem['lecture'].replace("\n", "\\n")
    return lecture


def get_solution_text(problem):
    # \\n: GPT-3 can generate the solution with more tokens
    solution = problem['solution'].replace("\n", "\\n")
    return solution


def create_one_example_chatbot(format, question, context, choice, answer, lecture, solution, test_example=True):

    input_format, output_format = format.split("-")

    ## Inputs
    if input_format == "CQM":
        input = f"Context: {context}\nQuestion: {question}\nOptions: {choice}\n"
    elif input_format == "QCM":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\n"
    # upper bound experiment
    elif input_format == "QCML":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture}\n"
    elif input_format == "QCME":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {solution}\n"
    elif input_format == "QCMLE":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture} {solution}\n"

    elif input_format == "QCLM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture}\nOptions: {choice}\n"
    elif input_format == "QCEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {solution}\nOptions: {choice}\n"
    elif input_format == "QCLEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture} {solution}\nOptions: {choice}\n"

    # Outputs
    if test_example:
        output = "Answer:"
    elif output_format == 'A':
        output = f"Answer: The answer is {answer}."

    elif output_format == 'AL':
        output = f"Answer: The answer is {answer}. BECAUSE: {solution}"
    elif output_format == 'AE':
        output = f"Answer: The answer is {answer}. BECAUSE: {lecture}"
    elif output_format == 'ALE':
        output = f"Answer: The answer is {answer}. BECAUSE: {lecture} {solution}"
    elif output_format == 'AEL':
        output = f"Answer: The answer is {answer}. BECAUSE: {solution} {lecture}"

    elif output_format == 'LA':
        output = f"Answer: {lecture} The answer is {answer}."
    elif output_format == 'EA':
        output = f"Answer: {solution} The answer is {answer}."
    elif output_format == 'LEA':
        output = f"Answer: {lecture} {solution} The answer is {answer}."
    elif output_format == 'ELA':
        output = f"Answer: {solution} {lecture} The answer is {answer}."
    elif output_format == 'LEPA':
        output = ''
        if len(lecture.strip()) > 0:
            output += f"LECTURE: {lecture}\n"
        if len(solution.strip()) > 0:
            output += f"SOLUTION: {solution}\n"
        output += '###\n'
        output += f"ANSWER: {answer}."

    input = input.replace("  ", " ").strip()
    output = output.replace("  ", " ").strip()
    if input.endswith("BECAUSE:"):
        input = input.replace("BECAUSE:", "").strip()
    if output.endswith("BECAUSE:"):
        output = output.replace("BECAUSE:", "").strip()
    return input, output


def create_one_example(format, question, context, choice, answer, lecture, solution, test_example=True):

    input_format, output_format = format.split("-")

    ## Inputs
    if input_format == "CQM":
        input = f"Context: {context}\nQuestion: {question}\nOptions: {choice}\n"
    elif input_format == "QCM":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\n"
    # upper bound experiment
    elif input_format == "QCML":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture}\n"
    elif input_format == "QCME":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {solution}\n"
    elif input_format == "QCMLE":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture} {solution}\n"

    elif input_format == "QCLM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture}\nOptions: {choice}\n"
    elif input_format == "QCEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {solution}\nOptions: {choice}\n"
    elif input_format == "QCLEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture} {solution}\nOptions: {choice}\n"

    # Outputs
    if test_example:
        output = "Answer:"
    elif output_format == 'A':
        output = f"Answer: The answer is {answer}."

    elif output_format == 'AL':
        output = f"Answer: The answer is {answer}. BECAUSE: {solution}"
    elif output_format == 'AE':
        output = f"Answer: The answer is {answer}. BECAUSE: {lecture}"
    elif output_format == 'ALE':
        output = f"Answer: The answer is {answer}. BECAUSE: {lecture} {solution}"
    elif output_format == 'AEL':
        output = f"Answer: The answer is {answer}. BECAUSE: {solution} {lecture}"

    elif output_format == 'LA':
        output = f"Answer: {lecture} The answer is {answer}."
    elif output_format == 'EA':
        output = f"Answer: {solution} The answer is {answer}."
    elif output_format == 'LEA':
        output = f"Answer: {lecture} {solution} The answer is {answer}."
    elif output_format == 'ELA':
        output = f"Answer: {solution} {lecture} The answer is {answer}."

    text = input + output
    text = text.replace("  ", " ").strip()
    if text.endswith("BECAUSE:"):
        text = text.replace("BECAUSE:", "").strip()
    return text



def create_one_example_gpt4(format, question, context, choice, answer, lecture, solution, test_example=True):

    input_format, output_format = format.split("-")

    ## Inputs
    if input_format == "CQM":
        input = f"Context: {context}\nQuestion: {question}\nOptions: {choice}\n"
    elif input_format == "QCM":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\n"
    # upper bound experiment
    elif input_format == "QCML":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture}\n"
    elif input_format == "QCME":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {solution}\n"
    elif input_format == "QCMLE":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture} {solution}\n"

    elif input_format == "QCLM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture}\nOptions: {choice}\n"
    elif input_format == "QCEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {solution}\nOptions: {choice}\n"
    elif input_format == "QCLEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture} {solution}\nOptions: {choice}\n"

    # Outputs
    if test_example:
        output = "Answer:"
    elif output_format == 'A':
        output = f"Answer: The answer is {answer}."

    elif output_format == 'AL':
        output = f"Answer: The answer is {answer}. BECAUSE: {solution}"
    elif output_format == 'AE':
        output = f"Answer: The answer is {answer}. BECAUSE: {lecture}"
    elif output_format == 'ALE':
        output = f"Answer: The answer is {answer}. BECAUSE: {lecture} {solution}"
    elif output_format == 'AEL':
        output = f"Answer: The answer is {answer}. BECAUSE: {solution} {lecture}"

    elif output_format == 'LA':
        output = f"Answer: {lecture} The answer is {answer}."
    elif output_format == 'EA':
        output = f"Answer: {solution} The answer is {answer}."
    elif output_format == 'LEA':
        output = f"Answer: {lecture} {solution} The answer is {answer}."
    elif output_format == 'ELA':
        output = f"Answer: {solution} {lecture} The answer is {answer}."

    input = input.replace("  ", " ").strip()
    output = output.replace("  ", " ").strip()
    if output.endswith("BECAUSE:"):
        output = output.replace("BECAUSE:", "").strip()

    user_prompt = {"role": "user", "content": f"Can you explain {input}?"}
    assistant_prompt = {"role": "assistant", "content": f"{output}"}

    return user_prompt, assistant_prompt


def build_prompt_chatbot(problems, shot_qids, prompt_format, use_caption=False, options=["A", "B", "C", "D", "E"], is_test=False):
    examples = {}

    for qid in shot_qids:
        question = get_question_text(problems[qid])
        context = get_context_text(problems[qid], use_caption)
        choice = get_choice_text(problems[qid], options)
        answer = get_answer(problems[qid], options)
        lecture = get_lecture_text(problems[qid]).replace('\\n', '\n')
        solution = get_solution_text(problems[qid]).replace('\\n', '\n')

        train_example = create_one_example_chatbot(prompt_format,
                                           question,
                                           context,
                                           choice,
                                           answer,
                                           lecture,
                                           solution,
                                           test_example=is_test)
        examples[qid] = train_example
    return examples


def build_prompt(problems, shot_qids, test_qid, args):

    examples = []

    # n-shot training examples
    for qid in shot_qids:
        question = get_question_text(problems[qid])
        context = get_context_text(problems[qid], args.use_caption)
        choice = get_choice_text(problems[qid], args.options)
        answer = get_answer(problems[qid], args.options)
        lecture = get_lecture_text(problems[qid])
        solution = get_solution_text(problems[qid])

        train_example = create_one_example(args.prompt_format,
                                           question,
                                           context,
                                           choice,
                                           answer,
                                           lecture,
                                           solution,
                                           test_example=False)
        examples.append(train_example)

    # test example
    question = get_question_text(problems[test_qid])
    context = get_context_text(problems[test_qid], args.use_caption)
    choice = get_choice_text(problems[test_qid], args.options)
    answer = get_answer(problems[test_qid], args.options)
    lecture = get_lecture_text(problems[test_qid])
    solution = get_solution_text(problems[test_qid])

    test_example = create_one_example(args.prompt_format,
                                      question,
                                      context,
                                      choice,
                                      answer,
                                      lecture,
                                      solution,
                                      test_example=True)
    examples.append(test_example)

    # create the prompt input
    prompt_input = '\n\n'.join(examples)

    return prompt_input


def build_prompt_gpt4(problems, shot_qids, test_qid, args):

    prompt_array = [{"role": "system", "content": "You are a helpful assistant."}]

    # n-shot training examples
    for qid in shot_qids:
        question = get_question_text(problems[qid])
        context = get_context_text(problems[qid], args.use_caption)
        choice = get_choice_text(problems[qid], args.options)
        answer = get_answer(problems[qid], args.options)
        lecture = get_lecture_text(problems[qid])
        solution = get_solution_text(problems[qid])

        user_prompt, assistant_prompt = create_one_example_gpt4(args.prompt_format,
                                           question,
                                           context,
                                           choice,
                                           answer,
                                           lecture,
                                           solution,
                                           test_example=False)
        prompt_array.append(user_prompt)
        prompt_array.append(assistant_prompt)

    # test example
    question = get_question_text(problems[test_qid])
    context = get_context_text(problems[test_qid], args.use_caption)
    choice = get_choice_text(problems[test_qid], args.options)
    answer = get_answer(problems[test_qid], args.options)
    lecture = get_lecture_text(problems[test_qid])
    solution = get_solution_text(problems[test_qid])

    user_prompt, assistant_prompt = create_one_example_gpt4(args.prompt_format,
                                      question,
                                      context,
                                      choice,
                                      answer,
                                      lecture,
                                      solution,
                                      test_example=True)
    prompt_array.append(user_prompt)
    prompt_array.append(assistant_prompt)

    return prompt_array


def convert_to_llava(base_dir, split, prompt_format="QCM-LEPA"):
    split_indices = json.load(open(os.path.join(base_dir, "pid_splits.json")))[split]
    problems = json.load(open(os.path.join(base_dir, "problems.json")))

    split_problems = build_prompt_chatbot(
        problems, split_indices, prompt_format,
        use_caption=False, is_test=False)

    target_format = []
    for prob_id, (input, output) in split_problems.items():
        if input.startswith('Question: '):
            input = input.replace('Question: ', '')
        if output.startswith('Answer: '):
            output = output.replace('Answer: ', '')

        raw_prob_data = problems[prob_id]
        if raw_prob_data['image'] is None:
            target_format.append({
                "id": prob_id,
                "conversations": [
                    {'from': 'human', 'value': f"{input}"},
                    {'from': 'gpt', 'value': f"{output}"},
                ],
            })

        else:
            target_format.append({
                "id": prob_id,
                "image": os.path.join(prob_id, raw_prob_data['image']),
                "conversations": [
                    {'from': 'human', 'value': f"{input}\n<image>"},
                    {'from': 'gpt', 'value': f"{output}"},
                ],
            })

    print(f'Number of samples: {len(target_format)}')

    with open(os.path.join(base_dir, f"llava_{split}_{prompt_format}.json"), "w") as f:
        json.dump(target_format, f, indent=2)


def convert_to_jsonl(base_dir, split, prompt_format="QCM-LEPA"):
    split_indices = json.load(open(os.path.join(base_dir, "pid_splits.json")))[split]
    problems = json.load(open(os.path.join(base_dir, "problems.json")))

    split_problems = build_prompt_chatbot(
        problems, split_indices, prompt_format,
        use_caption=False, is_test=False)

    writer = open(os.path.join(base_dir, f"scienceqa_{split}_{prompt_format}.jsonl"), "w")
    for prob_id, (input, output) in split_problems.items():
        if input.startswith('Question: '):
            input = input.replace('Question: ', '')
        if output.startswith('Answer: '):
            output = output.replace('Answer: ', '')

        raw_prob_data = problems[prob_id]
        if raw_prob_data['image'] is None:
            data = {
                "id": prob_id,
                "instruction": f"{input}",
                "output": f"{output}",
            }

        else:
            data = {
                "id": prob_id,
                "image": os.path.join(prob_id, raw_prob_data['image']),
                "instruction": f"{input}\n<image>",
                "output": f"{output}",
            }
        writer.write(json.dumps(data) + '\n')
    writer.close()


def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)