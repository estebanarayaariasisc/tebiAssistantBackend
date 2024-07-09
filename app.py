from app import create_app
from app.nlp import start_up_training_nodel

app = create_app()

def main():
    #start_up_training_nodel()
    app.run(debug=True)
if __name__ == '__main__':
    main()
    
    
