import re
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

def get_wishes():
    df = pd.read_csv('wishes.csv')
    wishes_data = []
    for _, row in df.iterrows():
        wishes_data.append({'name': row.iloc[1], 'wish': row.iloc[2]})
    return wishes_data

def get_tokenized_wish():
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    df = pd.read_csv('wishes.csv')
    wishes_data = ""

    # Define a tokenizer that selects tokens of alphanumeric characters, i.e., words
    tokenizer = RegexpTokenizer(r'\w+')

    # Get the English stop words list
    stop_words = set(stopwords.words('english'))

    for _, row in df.iterrows():
        wish_text = row.iloc[2]

        # Remove special characters and emoji
        wish_text = emoji_pattern.sub(r'', wish_text)

        # Tokenize the wish text into words
        tokens = tokenizer.tokenize(wish_text)

        # Remove stop words and non-alphabetic tokens
        words = ["".join(c for c in word if c.isalnum()).lower() for word in tokens if word.lower() not in stop_words]

        # Join the words back into one string separated by space and append it to the total data
        wishes_data += " ".join(words) + " "
    
    wishes_data = wishes_data \
        .replace(" lu", "") \
        .replace(" aku", "") \
        .replace(" di", "") \
        .replace(" yg", "")

    return wishes_data


def get_word_cloud(tokenized_words):

    # Create the word cloud object
    wordcloud = WordCloud(
        width=1600, height=800, background_color ='#F8F6EB', colormap="copper").generate(tokenized_words)
    wordcloud.to_file("./static/images/wordcloud.png")

    # Display the word cloud using matplotlib
    # plt.figure(figsize=(15, 8))
    # plt.imshow(wordcloud, interpolation='bilinear')
    # plt.axis('off')  # Hide the axes
    # plt.show()

    # This code would display a word cloud of the combined text from the wishes list.
    # You can adjust the width, height, and background_color as needed for your specific use case.

def get_word_frequencies(tokenized_words):

    count_dict = {}
    for word in tokenized_words.split(" "):
        count_dict[word] = count_dict.get(word, 0) + 1

    encouragement_words = ["happy", "best", "bless", "hope", "together", "love", "pray", "sukses", "happiest", "future", "grow", "opportunity", "sayang"]
    count_dict = {k: v for k, v in count_dict.items() if k in encouragement_words}
    count_dict = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
    words, freqs = zip(*count_dict)
    # sum_freqs = sum(freqs)
    # freqs = [round((100*freq)/sum_freqs, 2) for freq in freqs]

    # Create the barplot
    plt.figure(figsize=(12, 8))
    # colors = sns.color_palette('muted', n_colors=len(count_dict))  # Color palette for visual appeal
    colors = sns.color_palette()
    ax = sns.barplot(x=freqs, y=words, hue=words, legend=False, palette=colors)

    # Add annotations to each bar
    for i, v in enumerate(freqs):
        plt.text(v, i, " "+str(v), color='#777B7E', va='center', fontweight='bold')

    # Set the figure's background color
    plt.gcf().set_facecolor('#F8F6EB')

    # Set the axes' background color
    ax.set_facecolor('#F8F6EB')
    ax.set_yticklabels(ax.get_yticklabels(), color="#48494B")
    ax.set_xticklabels([])
    ax.xaxis.set_ticks_position('none')
    

    # Set the title and labels
    plt.title('', fontsize=18)
    plt.xlabel('', fontsize=14)
    plt.ylabel('', fontsize=14)

    # Remove the left spine for aesthetics
    sns.despine(left=True, bottom=True)

    # Save the figure to a file
    plt.savefig('static/images/word_freqs.png', bbox_inches='tight', dpi=500)

    plt.close()



if __name__ == "__main__":
    wishes_data = get_tokenized_wish()
    get_word_cloud(wishes_data)
    get_word_frequencies(wishes_data)