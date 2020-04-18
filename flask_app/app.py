# flask_web/app.py

from flask import Flask, render_template, url_for, request
from form import SubmitForm
# import neuralnet

app = Flask(__name__)
app.debug = True
app.config['SECRET_KEY'] = 'lyricsgenv1.0'

@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    form = SubmitForm()
    if form.validate_on_submit():
        print('1')
        # model = neuralnet.createModel()
        # generatedText = neuralnet.generate_text(model, form.lyrics.data, form.characters.data, float(form.temperature.data/10))
        # print(generatedText)
        # form.lyrics.data = generatedText
#     if request.method == 'POST':
#         temperature = request.form.get('temperature')  # access the data inside 
#         lyrics = request.form.get('lyrics')

    return render_template('home.html', form=form)


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/settings')
def settings():
        return render_template('settings.html')


if( __name__ == '__main__'):
    app.run(debug=True, host='0.0.0.0')