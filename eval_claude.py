import datetime
import json
import os
import re
import time

import anthropic
from matplotlib import pyplot as plt

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

answer_template = """

Answer: ##X##

(Note: 'X' is the answer choosed from A, B, C and D)
"""

class Evaluator:
    def __init__(self, benchmark_json, model_name, bench_name):
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



    def chat_completion(self, question):
        while True:
            try:
                client = anthropic.Anthropic(api_key="sk-ant-xxx")
                gpt_message = client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=500,
                    messages=[
                        {"role": "user", "content": f"Now you should choose the correct answer regarding the question and options. You should ONLY return the correct answer by following this template: {self.answer_template}\n\n You **MUST** follow the template to answer!!! Other format answers will be strongly rejected!"},
                        {"role": "assistant", "content": "Sure, I will choose the correct answer by following the requirements strictly."},
                        {"role": "user", "content": f"{question}"},
                    ],
                    temperature=0.1
                ).content[0].text
                return gpt_message

            except Exception as e:
                print("💤 (Lazy sleep mode)  Error: ", e)
                time.sleep(10)


    def extract_answer_from_source(self, source_answer) -> str:
        match = re.search(r'##(.*?)##', source_answer)
        print(f"[{datetime.datetime.now()}] [GPT] Source answer: ", source_answer)

        if match and match.group(1) in ["A", "B", "C", "D"]:
            return match.group(1)
        elif source_answer[8] in ["A", "B", "C", "D"]:
            return source_answer[8]
        else:
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
        plt.xlim(0, 302)
        plt.xlabel('Evaluation cases')
        plt.ylabel('Accuracy')
        plt.title('Evaluation Process Curve')
        plt.grid(True)
        plt.savefig(f'../eval_results/{self.model}_convergence_accuracy_plot.png', format='png')
        plt.show()

        with open(f"../eval_results/{self.model}_eval_{self.bench_name}_status.json", mode='w', encoding="utf-8") as j:
            json.dump(save_all_json_obj, j, indent=4)


    def evaluate(self):
        for idx, instance in enumerate(self.benchmark):
            if idx <= 488:
                print("idx: ", idx)
                with open("/Users/mellen/Desktop/MeRE-ms/project/eval_results/claude-3-opus-20240229_eval___FCO775___status.json", mode='r') as g:
                    self.status = json.load(g)
                    self.score = int(self.status['current_QAs'] * self.status['score_process'][-1])
                continue

            id_num = instance['id']
            conv = instance['conversations']
            human = conv[0]['value']
            gpt = conv[1]['value']

            question = human
            truth = gpt

            predict = self.chat_completion(question)
            pred_choice = self.extract_answer_from_source(predict)

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
            print(f"- - - - - - - - -"*3)
            print(f"👦 [    Human   ]: <{human}>")
            print(f"👦 [    Truth   ]: <{gpt}>\n")
            print(f"🤖 [     GPT    ]: <{pred_choice}>")
            print(f"🤖 [ Prediction ]: <{predict}>")
            print(f"   [    Status  ]: <{'✅' if correct else '❌'}>")
            print(f"- - - - - - - - -"*3)
            print(f"👣 [{datetime.datetime.now()}]")
            print(f"- - - - - - - - -"*3)
            print(f"🍏 [{idx+1}/{len(self.benchmark)}] Test Accuracy (%): {self.status['final_score']}\n\n\n")
            time.sleep(2)


if __name__ == '__main__':

    with open("./data/eval_data.json", mode='r') as f:
        benchmark = json.load(f)
    print("Total cases: ", len(benchmark))

    evaluator = Evaluator(benchmark_json=benchmark, model_name='claude-3-opus-20240229', bench_name="__FCO775__temp0.1")
    evaluator.evaluate()