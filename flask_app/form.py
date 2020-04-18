from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, IntegerField, TextAreaField
from wtforms.validators import DataRequired
from wtforms.widgets import TextArea

class SubmitForm(FlaskForm):
    
    temperature = IntegerField('temperature', validators=[DataRequired()])
    characters = StringField('characters', validators=[DataRequired()])
    lyrics = StringField('lyrics', widget=TextArea(), validators=[DataRequired()])

    submit = SubmitField('Generate')