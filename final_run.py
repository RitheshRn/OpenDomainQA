import tkinter
import os
import subprocess
import random
import jsonlines
import webbrowser
from nltk import sent_tokenize
from tkinter import StringVar
from tkinter import *

window = tkinter.Tk()

# to rename the title of the window
window.title("Open Domain QA")
window.geometry('800x400')

top_frame = tkinter.Frame(window).pack()

def good_example_1():
    output = subprocess.check_output('python3 bert_joint.py sample_best1.jsonl', shell=True)
    root = Tk()
    S = Scrollbar(root)
    T = Text(root)
    S.pack(side=RIGHT, fill=Y)
    T.pack(side=LEFT, expand=True, fill=BOTH)
    S.config(command=T.yview)
    T.config(yscrollcommand=S.set)
    T.insert(END, output)    

def good_example_2():
    output = subprocess.check_output('python3 bert_joint.py sample_best2.jsonl', shell=True)
    root = Tk()
    S = Scrollbar(root)
    T = Text(root)
    S.pack(side=RIGHT, fill=Y)
    T.pack(side=LEFT, expand=True, fill=BOTH)
    S.config(command=T.yview)
    T.config(yscrollcommand=S.set)
    T.insert(END, output)
    
def bad_example_1():
    output = subprocess.check_output('python3 bert_joint.py sample_bad1.jsonl', shell=True)
#     lbl['text'] = output.strip()
#     window1 = tkinter.Tk()
#     lbll = tkinter.Label(window1, text=output.strip())
#     lbll.pack()
    root = Tk()
    S = Scrollbar(root)
    T = Text(root)
    S.pack(side=RIGHT, fill=Y)
    T.pack(side=LEFT, expand=True, fill=BOTH)
    S.config(command=T.yview)
    T.config(yscrollcommand=S.set)
    T.insert(END, output)

    
def bad_example_2():
    output = subprocess.check_output('python3 bert_joint.py sample_bad2.jsonl', shell=True)
    root = Tk()
    S = Scrollbar(root)
    T = Text(root)
    S.pack(side=RIGHT, fill=Y)
    T.pack(side=LEFT, expand=True, fill=BOTH)
    S.config(command=T.yview)
    T.config(yscrollcommand=S.set)
    T.insert(END, output)

def custom_input():
    wiki_pages_train = []
    j = [random.randint(1, 1000) for i in range(10)]
    with jsonlines.open('simplified-nq-train.jsonl','r') as f:
        c = 1
        for obj in f:
            if c in set(j):
                wiki_pages_train.append(obj)
            c += 1

            if len(wiki_pages_train) == 10:
                break
    
    wiki_names = [wiki_pages_train[i]['document_text'].split('<')[0] for i in range(10)]
    options = [wiki_pages_train[i]['document_url'] for i in range(10)]
    
    wiki_page_dict = {wiki_names[i]:options[i] for i in range(10)}    
    variable = StringVar(window)
    variable.set("Choose Wikipedia article") 
    w = OptionMenu(window, variable, *wiki_names)
    w.pack()   
    

    def ok():
        t = variable.get()
        with jsonlines.open('Intermediate_jsonl.jsonl', mode='w') as f:
            f.write(wiki_pages_train[options.index(wiki_page_dict[t])])
        webbrowser.open(wiki_page_dict[t])
        
    button = Button(window, text="OK", command=ok)
    button.pack()
    
def custom_input_question():
    
    def retrieve_input():
        question=textBox.get("1.0","end-1c")
        if question[-1] == '?':
            question = question[:-1]
        q_set = set(question.split(' '))
#         stop_words = {'is', 'in', 'a', 'an', 'the', 'for', 'he', 'that', 'she', 'this'}
#         q_set = q_set.difference(stop_words)
        with jsonlines.open('Intermediate_jsonl.jsonl','r') as f:
            for obj in f:
                datapoints = obj

        q1 = datapoints['document_text']
        q2 = sent_tokenize(q1)
        q3 = []
        for sent in q2:
            q3.append(sent.split(' '))

        start_end = []
        c = 1
        for idx, i in enumerate(q3):
            flag = True
            if len(q_set.intersection(set(i))) > 1 and flag:
                tmp = {'start_token': 0, 'top_level': True, 'end_token': 0}
                tmp['start_token'] = c
                tmp['end_token'] = c + len(i)-1
                start_end.append(tmp)
                flag = False
            c += len(i)
            
        # write in json fromat 
        z = {}
        z['example_id'] = datapoints['example_id']
        z['question_text'] = question
        z['document_text'] = datapoints['document_text']
        z['long_answer_candidates'] = start_end

        with jsonlines.open('sample_custom.jsonl', mode='w') as f:
            f.write(z)
        try:
            output = subprocess.check_output('python3 bert_joint.py sample_custom.jsonl', shell=True)
        except:
            output = "NO suitable answer found!"
        root = Tk()
        S = Scrollbar(root)
        T = Text(root)
        S.pack(side=RIGHT, fill=Y)
        T.pack(side=LEFT, expand=True, fill=BOTH)
        S.config(command=T.yview)
        T.config(yscrollcommand=S.set)
        T.insert(END, output)   
        
        
    textBox=Text(window, height=3, width=50)
    textBox.pack()
    buttonCommit=Button(window, height=1, width=10, text="Commit", 
                    command=lambda: retrieve_input())
    buttonCommit.pack()
    
    
lbl1 = tkinter.Label(window, text='Open Domain Question Answering using Natural Questions')    
lbl = tkinter.Label(window, text='')

btn1 = tkinter.Button(top_frame, text = "Good Example 1", fg = "blue", command = good_example_1).place(x=50,y=150)
btn2 = tkinter.Button(top_frame, text = "Good Example 2", fg = "blue", command = good_example_2).place(x=50,y=200)
btn3 = tkinter.Button(top_frame, text = "Bad Example 1", fg = "red", command = bad_example_1).place(x=650,y=150)
btn4 = tkinter.Button(top_frame, text = "Bad Example 2", fg = "red", command = bad_example_2).place(x=650,y=200)
btn5 = tkinter.Button(top_frame, text = "      Custom Input     ", fg = "green", command = custom_input).place(x=350,y=300)
btn6 = tkinter.Button(top_frame, text = "  Custom Input Question  ", fg = "green", command = custom_input_question).place(x=336,y=350)


lbl1.pack()
lbl.pack()

window.mainloop()