from math_agent import MathAgent

math_agent = MathAgent()

while True:
    print("=================================")
    print("\nUser:")
    user_input = input()
    response = math_agent.run(user_input)
    print("\nAI:")
    print(response)
    print("\n\n")

