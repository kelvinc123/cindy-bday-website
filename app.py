from flask import Flask, render_template
import wishes_processor

app = Flask(__name__)

@app.route('/')
def index():
    wishes = wishes_processor.get_wishes()
    return render_template('index.html', wishes=wishes)

if __name__ == '__main__':
    tokenized_words = wishes_processor.get_tokenized_wish()
    wishes_processor.get_word_cloud(tokenized_words=tokenized_words)
    wishes_processor.get_word_frequencies(tokenized_words=tokenized_words)
    app.run()