import tkinter as tk#creating gui-graphical user interface
from tkinter import scrolledtext,ttk,messagebox
import random 
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import webbrowser
import datetime
class ChatbotGUI:
    #defining the chatbot GUI class
    def __init__(self,master):#master is main window
        self.master=master#stores main window inside class
        self.setup_gui()#calling the function that is creates all gui compoments,its builds the chat bot window
        self.load_chatbot_data()#another function chatbot model
        self.conversation_history=[]
    def setup_gui(self):#create chatbot interface
        self.master.title("GYM-Bot")
        self.master.geometry("600x500")#width,height
        self.master.configure(bg="#244923")

        style=ttk.Style()#create style object
        style.theme_use("clam")

        main_frame=ttk.Frame(self.master,padding="10")#create frame ,holds other widgets.
        main_frame.pack(fill=tk.BOTH,expand=True)#fill horizontally and vertically ,expand--window resized
        #chat history text Box
        self.chat_history=scrolledtext.ScrolledText(
            main_frame,wrap=tk.WORD,width=60,height=25,font=("Arial",10))
        self.chat_history.pack(padx=10,pady=10,fill=tk.BOTH,expand=True)
        self.chat_history.config(state=tk.DISABLED)
        #input and button Frames
        input_frame=ttk.Frame(main_frame)
        input_frame.pack(fill=tk.X,pady=5)
        self.user_input=ttk.Entry(input_frame,width=50,font=("Arial",10))
        self.user_input.pack(side=tk.LEFT,padx=(0,5),expand=True,fill=tk.X)
        self.user_input.bind("<Return>", lambda event:self.send_message())

        self.send_button=ttk.Button(
            input_frame,text="Send",command=self.send_message
        )
        self.send_button.pack(side=tk.RIGHT)
        #button Frame for clear ,save and helpt
        button_frame=ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X,pady=5)
        self.clear_button=ttk.Button(
            button_frame,text="Clear Chat",command=self.clear_chat
        )
        self.clear_button.pack(side=tk.LEFT,padx=(0,5))
        self.save_button=ttk.Button(
            button_frame,text="Save Chat",command=self.save_chat
        )
        self.save_button.pack(side=tk.LEFT)
        self.help_button=ttk.Button(button_frame,text="Help",command=self.show_help)
        self.help_button.pack(side=tk.RIGHT)
        #loading chatbot Data
    def load_chatbot_data(self):
        self.lemmatizer=WordNetLemmatizer()
        self.intents=json.loads(
            open(r'C:\Users\raj93\OneDrive\Desktop\nlp_bot\data\intents.json').read()

        )    
        self.words=pickle.load(open("words.pkl","rb"))
        self.classes=pickle.load(open("classes.pkl","rb"))
        self.model=load_model("chatbot_model.h5")
    #sending and receiving Messages
    def send_message(self):
        user_message=self.user_input.get()
        self.user_input.delete(0,tk.END)
        if user_message:
            self.update_chat_history(f"You: {user_message}")
            bot_response=self.get_bot_response(user_message)
            self.update_chat_history(f"Bot: {bot_response}")
            self.conversation_history.append((user_message,bot_response))
    #upadatin chat history
    def update_chat_history(self,message):
        self.chat_history.config(state=tk.NORMAL)
        self.chat_history.insert(tk.END,message+"\n\n")
        self.chat_history.see(tk.END)
        self.chat_history.config(state=tk.DISABLED)
    #GETTING BOT RESPONSE
    def get_bot_response(self,user_message):
        if user_message.lower()in ["exit","quit","bye"]:
            return "Goodbye! have a nice day!"
        elif user_message.lower().startswith("search "):
            query=user_message[7:]
            webbrowser.open(f"https://www.google.com/search?q={query}")
            return f"I have opened a web search for 'query'."
        elif user_message.lower()=="time":
            return (
                f"The current time is {datetime.datetime.now().strftime('%H:%M:%S')}.")
        
        else:
            ints=self.predict_class(user_message)
            return self.get_response(ints)
    #cleaning up sentence and bag of words
    def clean_up_sentence(self,sentence):
        return [
            self.lemmatizer.lemmatize(word.lower())for word in nltk.word_tokenize(sentence)
        ]
    def bag_of_words(self,sentence):
        sentence_words=self.clean_up_sentence(sentence)
        bag=[1 if word in sentence_words else 0 for word in self.words]
        return np.array(bag)
    #predicting class and getting response
    def predict_class(self,sentence):
        bow =self.bag_of_words(sentence)
        res=self.model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD=0.25
        results=[[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
        results.sort(key=lambda x:x[1],reverse=True)
        return [
            {"intent":self.classes[r[0]],"probability":str(r[1])} for r in results
        ]
    def get_response(self,intents_list):
        if not intents_list:
            return "i m not sure how to respond to that .can you please rephrase your question?"
        tag=intents_list[0]["intent"]
        for intent in self.intents["intents"]:
            if intent["tag"]==tag:
                return random.choice(intent["responses"])
        return "i am sorry ,i don't have a specific responses for that .can you try asking something else?"
    #cleaning
    def clear_chat(self):
        self.chat_history.config(state=tk.NORMAL)
        self.chat_history.delete(1.0,tk.END)
        self.chat_history.config(state=tk.DISABLED)
        self.conversation_history.clear()
    #save chat
    def save_chat(self):
        filename=(
            f"chat_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        with open (filename,"w") as  f:
            for user_msg,bot_msg in self.conversation_history:
                f.write(f"You: {user_msg}\n")
                f.write(f"Bot: {bot_msg}\n\n")
        messagebox.showinfo("Chat Saved",f"chat history has been saved to {filename}")
    #showing HElp
    def show_help(self):
        help_text="""
        Welcome to the GYM-Bot!

        Special Commands:
        -Type 'exit','quit',or'bye' to end the conversation.
        -Type 'search <query>' to open a web search.
        -Type 'time' to get the current time.
         Features:
         -Clear Chat:Clears the current conversation.
         -Save Chat:Save the conversation history to a fille.
         -Help:Shows this help message.

         Enjoy chatting!
            """
        messagebox.showinfo("Chatbot Help",help_text)

    #running the application
if __name__=="__main__":
    root=tk.Tk() 
    ChatbotGUI(root)
    root.mainloop()

