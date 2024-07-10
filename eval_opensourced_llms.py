import datetime
import json
import os
import re
import time

import openai
from matplotlib import pyplot as plt

# Set up the OpenAI API key
api_base_addr = "http://10.13.27.8:9000/v1"

openai.api_key = "EMPTY"
openai.api_base = api_base_addr

os.environ['OPENAI_API_KEY'] = "EMPTY"
os.environ['OPENAI_API_BASE'] = api_base_addr
os.environ['FASTCHAT_WORKER_API_EMBEDDING_BATCH_SIZE'] = "1"
answer_template = """

Answer: ##X##

(Note: 'X' is the answer choosed from A, B, C and D)
"""

class Evaluator:
    def __init__(self, benchmark_json, model_name, bench_name, temp=0., extractor=""):
        self.bench_name = bench_name
        self.benchmark = benchmark_json
        self.status = {
            "model": "",
            "final_score": 0.,
            "source_QA": [],
            "correct_answers": [],
            "wrong_answers": [],
            "score_process": [],
            "total_QAs": 0,
            "current_QAs": 0,
        }
        self.score = 0
        self.answer_template = answer_template
        self.model = model_name
        self.temp = temp
        self.extractor = extractor

    def davinci_complete_prompt(self, prompt, temperature=0.7, max_tokens=100):
        """
        This function is dropped in test model list. Only consider 'chat_completion' method.
        :param prompt:
        :param temperature:
        :param max_tokens:
        :return:
        """
        try:
            response = openai.Completion.create(
                engine=self.model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].text.strip()
        except Exception as e:
            return str(e)


    def chat_completion(self, question):
        while True:
            try:
                completion = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": f"Now you should say 'True' or 'False' regarding the question. You should ONLY return the correct answer by following this template: {self.answer_template}\n\n You **MUST** follow the template to answer!!! Other format answers will be strongly rejected!"},
                        {"role": "user", "content": f"{question}"},
                    ],
                    temperature=self.temp,
                    max_tokens=50
                )
                gpt_message = completion.choices[0].message.content

                _resp_ = openai.ChatCompletion.create(
                    model="vicuna-13b-v1.5",
                    temperature=0.1,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": f"""
Example:

Instruction: Extract the answer in the source output:

Content: "Based on the analysis, the answer is True, which says the growth is influenced by many factors."

Assistant: Answer: ##True## 

"""},
{"role": "user", "content": f"""
Instruction: Extract the answer in the source output:

Content: "{gpt_message}"

Assistant: 
"""},
                    ]
                )

                return _resp_.choices[0].message.content

            except Exception as e:
                print("💤 (Lazy sleep mode)  Error: ", e)
                time.sleep(10)


    def extract_answer_from_source(self, source_answer, qa_type) -> str:
        match = re.search(r'##(.*?)##', source_answer)
        print(f"[{datetime.datetime.now()}] [GPT] Source answer: ", source_answer)
        if qa_type == "TOF":
            if match and match.group(1) in ["True", "False"]:
                return match.group(1)
            elif ("True" in source_answer) or ("true" in source_answer) or ("TRUE" in source_answer):
                return "TRUE"
            elif ("False" in source_answer) or ("false" in source_answer) or ("FALSE" in source_answer):
                return "FALSE"
            else:
                return ""
        elif qa_type == "FCO":
            try:
                if match and match.group(1) in ["A", "B", "C", "D"]:
                    return match.group(1)
                elif source_answer[8] in ["A", "B", "C", "D"]:
                    return source_answer[8]
                else:
                    return ""
            except Exception as e:
                return ""

    def match(self, pred, truth):
        print(f"♻️   [{pred}] v.s. [{truth}]")
        if pred == truth:
            return True
        else:
            return False

    def save_report(self, save_all_json_obj):
        os.makedirs("../eval_results", exist_ok=True)
        data = self.status['score_process']

        plt.figure(figsize=(10, 6), dpi=300)
        plt.plot(data, color='#2d85ff', linewidth=2, linestyle='-', marker='o', markersize=4)
        convergence_value = data[-1]
        plt.axhline(y=convergence_value, color='#ff2e5a', linestyle='--')
        plt.text(len(data) - 1, convergence_value, f'Converged: {convergence_value}', color='#ff2e5a')
        plt.ylim(0.0, 1.0)
        # plt.xlim(0, 302)
        plt.xlabel('Evaluation cases')
        plt.ylabel('Accuracy')
        plt.title('Evaluation Process Curve')
        plt.grid(True)
        plt.savefig(f'../eval_results/{self.model}_FCO_convergence_accuracy_plot.png', format='png')
        plt.show()

        with open(f"../eval_results/{self.model}_FCO_eval_{self.bench_name}_status.json", mode='w', encoding="utf-8") as j:
            json.dump(save_all_json_obj, j, indent=4)


    def evaluate(self):
        for idx, instance in enumerate(self.benchmark):
            id_num = instance['id']
            conv = instance['conversations']
            human = conv[0]['value']
            gpt = conv[1]['value']

            question = human
            truth = gpt

            predict = self.chat_completion(question)

            if "FCO" in self.bench_name:
                qa_type = "FCO"
            elif "TOF" in self.bench_name:
                qa_type = "TOF"

            pred_choice = self.extract_answer_from_source(predict, qa_type=qa_type)
            self.status['source_QA'].append({
                "id": id_num,
                "question": human,
                "answer": gpt,
                "predicted": predict,
                "predicted_choice": pred_choice
            })

            # Correct answers
            if self.match(pred=pred_choice, truth=truth):
                correct = True
                self.score += 1
                self.status['correct_answers'].append({
                    "id": id_num,
                    "question": human,
                    "answer": gpt,
                    "predicted_choice": pred_choice
                })
            else:
                correct = False
                self.status['wrong_answers'].append({
                    "id": id_num,
                    "question": human,
                    "answer": gpt,
                    "predicted_choice": pred_choice
                })

            accuracy_proc = self.score/(idx+1)

            self.status['score_process'].append(accuracy_proc)
            self.status['total_QAs'] = len(self.benchmark)
            self.status['current_QAs'] = idx+1
            self.status['final_score'] = accuracy_proc

            self.save_report(self.status)
            print(f"- - - - - Selection - - - - -"*3)
            print(f"👦 [    Human   ]: <{human}>")
            print(f"👦 [    Truth   ]: <{gpt}>\n")
            print(f"🤖 [     GPT    ]: <{pred_choice}>")
            print(f"🤖 [ Prediction ]: <{predict}>")
            print(f"   [    Status  ]: <{'✅' if correct else '❌'}>")
            print(f"- - - - - - - - -"*3)
            print(f"👣 [{datetime.datetime.now()}]")
            print(f"- - - - - - - - -"*3)
            print(f"🍏 [{idx+1}/{len(self.benchmark)}] Test Accuracy (%): {self.status['final_score']}\n\n\n")


if __name__ == '__main__':
    with open("./data/eval_data.json", mode='r') as f:
        benchmark = json.load(f)

    evaluator = Evaluator(
        benchmark_json=benchmark,
        model_name='vicuna-13b-v1.5',
        extractor='vicuna-13b-v1.5',
        bench_name="baseline_temp0.9_FCO",
        temp=0.9
    )
    evaluator.evaluate()