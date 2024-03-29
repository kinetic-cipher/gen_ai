import tkinter as tk
from tkinter import scrolledtext
#from math_agent import MathAgent
from multi_modal_agent import MultiModalAgent

class AIChatUI:

    def clear_user_text_box(self,event):
        self.user_text_box.delete(1.0, tk.END)  # Delete all text in the text box
        self.user_text_box.mark_set("insert", "1.0")  # Move cursor to the beginning of the text box

    def clear_ai_text_box(self,event):
        self.ai_text_box.delete(1.0, tk.END)  # Delete all text in the text box
        self.ai_text_box.mark_set("insert", "1.0")  # Move cursor to the beginning of the text box

    def _setupUI(self, master):
        self.master = master
        master.title("AI Chat")

        # Make the window size larger
        master.geometry("600x800")

        # Set dark gray background color
        master.configure(bg='#404040')

        # Create user input text box
        self.user_label = tk.Label(master, text="User:", fg='#00bfff', bg='#404040', font=("Arial",18))
        self.user_label.pack()

        self.user_text_box = scrolledtext.ScrolledText(master, wrap=tk.WORD, width=80, height=10, bg='#1a1a1a',
                                                       fg='#00bfff', insertbackground='#66ff66', font=("Arial", 18))
        self.user_text_box.pack(pady=5)

        # Create AI response text box
        self.ai_label = tk.Label(master, text="AI:", fg='#00bfff', bg='#404040', font=("Arial",18))
        self.ai_label.pack()

        self.ai_text_box = scrolledtext.ScrolledText(master, wrap=tk.WORD, width=80, height=40, bg='#1a1a1a', fg='#0066cc',
                                                     font=("Arial",18))
        self.ai_text_box.pack(pady=5)

        # Set custom color scheme for the text boxes
        self.user_text_box.tag_config('user', foreground='#00bfff')
        self.ai_text_box.tag_config('ai', foreground='#00bfff')

        # Bind the 'Return' key to the process_query function
        self.user_text_box.bind("<Return>", self.process_query)
        self.user_text_box.bind("<Button-1>", self.clear_user_text_box)
        self.ai_text_box.bind("<Button-1>", self.clear_ai_text_box)


    def __init__(self, master):
        """
        initializer/constructor
        """
        self._setupUI(master)
        #self.math_agent = MathAgent()
        self.multi_modal_agent = MultiModalAgent()

    def process_query(self, event):

        # Clear the AI text box
        self.ai_text_box.delete(tk.END + "-2l", tk.END)

        # Get user input from the user text box
        user_query = self.user_text_box.get(tk.END + "-2l", tk.END).strip()

        # Display user's query in the user text box
        #self.user_text_box.insert(tk.END, "\n" + user_query, 'user')

        # Call the math agent
        #response = self.math_agent.run(user_query)
        response = self.multi_modal_agent.run(user_query)

        # Display AI's response in the AI text box
        self.ai_text_box.insert(tk.END, "\n"+response, 'ai')

        # Scroll the text boxes upward
        self.user_text_box.yview(tk.END)
        self.ai_text_box.yview(tk.END)

        # Clear the user's input in the user text box
        #self.user_text_box.delete(tk.END + "-2l", tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = AIChatUI(root)
    root.mainloop()

