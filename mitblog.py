import requests
import html
from html.parser import HTMLParser
import os
import random

MAX_BLOG_PAGES = 20
OUTPUT_COUNT = 20
MAX_OUTPUT_LENGTH = 512
MAX_SEQUENCE_LENGTH = 128
NUM_EPOCHS = 3
BATCH_SIZE = 3

class MyHTMLParser(HTMLParser):
    output = ""
    ignore = False

    def clear_output(self):
        self.output = ""
        self.ignore = False

    def handle_starttag(self, tag, attrs):
        if tag == "a" or tag == "b" or tag == "i" or tag=="p" or tag=="blockquote" or tag=="span":
            self.ignore = False
        else:
            self.ignore = True

    def handle_endtag(self, tag):
        if not(tag == "a" or tag == "b" or tag == "i" or tag=="p" or tag=="blockquote" or tag=="span"):
            self.ignore = False
        else:
            self.ignore = True

    def handle_data(self, data):
        if (not self.ignore) and (not data.isspace()) and (not data == '\n'):
            self.output = self.output + str(data) + " "

parser = MyHTMLParser()

def get_blogs():
    bloglinks = []
    listlink = 'https://mitadmissions.org/blogs/page/'
    listlinknum = 1
    bloglink = ""
    f = open("blogs.txt", "a", encoding="utf8")

    while listlinknum <= MAX_BLOG_PAGES:
        print(listlink + str(listlinknum))
        r = requests.get(listlink + str(listlinknum))
        contents = r.content.decode('utf-8').split('class="post-tease__h__link" href="')[1:]

        for content in contents:
            
            bloglink = content[0: content.find(">")-1]

            if bloglink in bloglinks:
                MyHTMLParser.close()
                return

            
            r = requests.get(bloglink).content.decode('utf-8')

            endarticle = r.find('<ol class="page__footnotes">')
            if endarticle == -1:
                endarticle = r.find('<div class="share-tools-mod">')

            startarticle = r.find('<div class="article__body js-hang-punc">')
            

            blog = "\n<|start of blog|>\n"
            
            parser.feed(r[startarticle+len('<div class="article__body js-hang-punc">'):endarticle])
            blog = blog + parser.output
            parser.clear_output()
            
            blog = blog + "\n<|end of blog|>"

            f.write(blog)
            bloglinks.append(bloglink)
        
        listlinknum = listlinknum + 1
    
    f.close()

if not os.path.exists('blogs.txt'):
    print("fetching blogs from web")
    get_blogs()

blogs = open("blogs.txt", "r", encoding="utf8").read()
blogs = blogs.split("<|end of blog|>")[1:]
random.shuffle(blogs)
print(type(blogs))
print(type(blogs[0]))

import keras_nlp
import tensorflow as tf
from tensorflow import keras


if os.path.exists('model.tf'):
    print('loading model')
    gpt2_lm = keras.models.load_model("model.tf", custom_objects={
        'GPT2Tokenizer': keras_nlp.models.GPT2Tokenizer,
        'GPT2Backbone': keras_nlp.models.GPT2Backbone
    })
else:
    
    print('creating model')

    # To speed up training and generation, we use preprocessor of length 128
    # instead of full length 1024.
    preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
        "gpt2_base_en",
        sequence_length=MAX_SEQUENCE_LENGTH,
    )

    gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset(
        "gpt2_base_en", preprocessor=preprocessor
    )


    # Linearly decaying learning rate.
    learning_rate = keras.optimizers.schedules.PolynomialDecay(
        5e-5,
        decay_steps=NUM_EPOCHS,
        end_learning_rate=0.0,
    )

    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    gpt2_lm.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=loss,
        weighted_metrics=["accuracy"],
        run_eagerly=False
    )

    print(blogs)
    gpt2_lm.fit(blogs, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
    gpt2_lm.summary()
    gpt2_lm.save("model.tf", save_format='tf',include_optimizer=False)



inputs = ['<|start of blog|>\n'] * OUTPUT_COUNT

output = gpt2_lm.generate(inputs, max_length=MAX_OUTPUT_LENGTH)
print("\nGPT-2 output:")
print(output)

f = open("fakeblogs.txt","a",encoding='utf8')
f.write('\n<|end of blog|>\n'.join(output))
f.close()
