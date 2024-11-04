from ttt_test import TicTacToe, Agent, load_model

model = load_model("tic_tac_toe_model_final.pth")
agent = Agent(model)
game = TicTacToe()

def play_human_vs_agent(agent, game):
    game.reset()
    while not game.check_winner():
        print(game.board[0:3])
        print(game.board[3:6])
        print(game.board[6:9])
        if game.current == 1:
            print("Your turn (X)")
            pos = int(input("Enter the pos (0-8): "))
            if game.move(pos):
                print("You made a move.")
            else:
                print("Invalid move. Try again.")
                continue
        else:
            print("Agent's turn (O)")
            valid_moves = game.get_valid_moves()
            action = agent.select_action(game.get_state(), valid_moves)
            if game.move(action):
                print("Agent made a move.")
            else:
                print("Agent made an invalid move. Something went wrong.")
                break

    
    print(game.board[0:3])
    print(game.board[3:6])
    print(game.board[6:9])
    winner = game.check_winner()
    if winner == 0:
        print("It's a tie!")
    elif winner == 1:
        print("Congratulations! You won!")
    else:
        print("The agent wins!")

play_human_vs_agent(agent, game)