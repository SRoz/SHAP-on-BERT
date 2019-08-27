import pandas as pd
import numpy as np

from IPython.display import display, Markdown


def display_shap_text(shap_values,  raw_batch, logits, tokenizer, n=0, m=None, has_actuals=True, threshold=0.5):
    df = {'toks' : [], 'vals' : []}
    
    if m=='pos':
        sentence_shap = shap_values[0][0][n]
    elif m=='neg':
        sentence_shap = -shap_values[1][0][n]
    else:
        sentence_shap = shap_values[0][0][n] - shap_values[1][0][n]
    
    for row, i in zip(sentence_shap, raw_batch[0][n]):

        tok = tokenizer.ids_to_tokens[int(i)]
        df['toks'] += [tok]
        df['vals'] += [row[i]]

        #print(str(tok) +" : "+ str(round(row.sum()*1000,2)))
    df = pd.DataFrame(df)

    df['vals'] = df.vals/(np.abs(df.vals).max())
    
    if has_actuals:
        labels = raw_batch[3]
        rating = 'good' if labels.numpy()[n] == 0 else 'bad'
        col = 'green' if labels.numpy()[n] == 0 else 'red'
        rating = f"<font color={col}>{rating}</font>" + " "
        display(Markdown(f"## Actual Rating: {rating}"))
    
    logits_ = logits[n]
    model_pred = 'good' if logits_.detach().numpy()[0] > logits_.detach().numpy()[1] else 'bad'
    model_col = 'green' if model_pred == 'good' else 'red'
    model_pred = f"<font color={model_col}>{model_pred}</font>" + " "
    
    display(Markdown(f"## Model prediction: {model_pred}"))
    display(Markdown(f"##### Logits (pos, neg): {logits_.detach().numpy()}"))

    string = ""
    for i in range(df.shape[0]-2):
        tok = df.iloc[i+1,0]
        val = df.iloc[i+1,1]
        
        col = "'red'" if val < 0 else "'green'"
        size = 3 + 5*np.abs(val)

        
        if np.abs(val) > threshold:
            string += (f"<font color={col}><font size={size}>{tok}</font>" + " ")
        else:
            string += (f"<font color=grey><font size={size}>{tok}</font>" + " ")
    
    display(Markdown(string))
    display(Markdown("---"))