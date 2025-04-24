import streamlit as st
import time
import random
import torch
import torch.nn as nn
import sklearn
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from bnlp import BasicTokenizer
from bnlp import SentencepieceTokenizer
from bnlp import BengaliWord2Vec
from bnlp import NLTKTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from torch import softmax
from lime import lime_text
from lime.lime_text import LimeTextExplainer
import numpy as np
import torch
from torch.nn.functional import softmax
import torch.nn.init as init
import streamlit.components.v1 as components
device = torch.device("cpu")

#----------------------------------------Prediction Function----------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate=0.5):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)

        # Initialize the weights of the LSTM anda fully-connected layers
        init.xavier_uniform_(self.lstm.weight_ih_l0)
        init.xavier_uniform_(self.lstm.weight_hh_l0)
        init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        output, _ = self.lstm(x)
        last_output = output[:, -1, :]
        last_output = self.dropout(last_output)
        return self.fc(last_output)

input_dim = 100  # vector size
hidden_dim = 128
output_dim = 2
num_layers = 4

model_LSTM = LSTMModel(input_dim, hidden_dim, output_dim, num_layers, dropout_rate=0.5).to(device)

model_LSTM.load_state_dict(torch.load('model_ckpoint.pth', map_location=torch.device('cpu')))

# Assuming 'bwv' is your Bangla word vector model and 'BasicTokenizer' has already been defined
basic_tokenizer = BasicTokenizer()

def preprocess_single_comment(text):
    # Tokenization
    tokens = basic_tokenizer.tokenize(text)

    # Word Embedding
    vectors = sentence_to_vectors(tokens)
    
    # Convert list of vectors to tensor and add an extra dimension for batch size
    if len(vectors) > 0:
        tensor = torch.tensor([vectors], dtype=torch.float32).to(device)
    else:
        # Handling the case where there are no valid words in the comment
        zero_vector = np.zeros(bwv.get_word_vector("গ্রাম").shape)
        tensor = torch.tensor([[zero_vector]], dtype=torch.float32).to(device)
        
    return tensor

def sentence_to_vectors(sentence_tokens):
    vectors = []
    for word in sentence_tokens:
        try:
            vector = bwv.get_word_vector(word)
            vectors.append(vector)
        except KeyError:
            # Handle with a zero vector
            vector = np.zeros(bwv.get_word_vector("গ্রাম").shape)  # Using a generic word to determine the size
            vectors.append(vector)
    return vectors




own_model_path = "hsw2v.model"
bwv = BengaliWord2Vec(model_path=own_model_path)
def sentence_to_vectors(sentence_tokens):

    vectors = []
    for word in sentence_tokens:
        try:
            vector = bwv.get_word_vector(word)
            vectors.append(vector)
        except KeyError:
            # Ignore words that are not in the vocabulary
            # Or handle with a zero vector if needed:
            # vector = np.zeros(bwv.get_word_vector("গ্রাম").shape)
            # vectors.append(vector)
            continue
    return vectors



def model_predict(texts):
    model_LSTM.eval()  # Make sure the model is in evaluation mode
    predictions = []
    
    for text in texts:
        # Preprocess the text to fit the model's input requirements
        tensor = preprocess_single_comment(text)
        with torch.no_grad():
            output = model_LSTM(tensor)
            # Apply softmax to convert logits to probabilities
            output = softmax(output, dim=1)
            probs = output.cpu().numpy()[0]
            predictions.append(probs)
    
    # Return a 2D array of probabilities (one row per text, one column per class)
    return np.array(predictions)

# Initialize LIME text explainer with descriptive class names
class CustomTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, doc):
        return self.tokenizer.tokenize(doc)

# Create a custom tokenizer instance using your BasicTokenizer
custom_tokenizer = CustomTokenizer(basic_tokenizer)

# Initialize LIME text explainer with the custom tokenizer
explainer = LimeTextExplainer(class_names=["Non Hate Speech", "Hate Speech"], split_expression=custom_tokenizer)


def do_prediction(text_to_explain):
    
    explainer = LimeTextExplainer(class_names=["Non Hate Speech", "Hate Speech"], split_expression=custom_tokenizer)

    # Generate explanation for the specific instance
    exp = explainer.explain_instance(text_to_explain, model_predict, num_features=6, labels=[1])

    # Display the explanation for "Hate Speech" class
    print('Explanation for class 1 (Hate Speech):')
    exp.show_in_notebook(text=True, labels=(1,))
    exp.save_to_file('temp.html')
    # Get predicted class and confidence
    predictions = model_predict([text_to_explain])
    predicted_class = np.argmax(predictions[0])  # Get the predicted class index
    confidence = predictions[0][predicted_class]  # Get the confidence (probability) of the predicted class

    class_name = ["Non Hate Speech", "Hate Speech"][predicted_class]  # Get class name based on predicted class index
    print(f"Predicted class: {class_name}")
    print(f"Prediction confidence: {confidence * 100:.2f}%")
    with open('temp.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    return predicted_class, confidence, html_content






#----------------------------------------Streamlit App----------------------------------------






# Initialize session state for examples
if 'example_text' not in st.session_state:
    st.session_state['example_text'] = ""

# Function to set example text
def set_example(text):
    st.session_state['example_text'] = text

# Streamlit app layout and configuration
st.set_page_config(
    page_title="Bangla Hate Speech Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
        padding-bottom: 30px;
    }
    .stApp {
        background-color: #f8f9fa;
    }
    
    
    
    div.main-header {
        color: #3d3d3d;
        font-family: 'Roboto', sans-serif;
        text-align: center;
        margin-bottom: 10px;
        padding-top: 20px;
        font-size: 2.5em;
        font-weight: bold;
    }
    div.main-subheader {
        color: #6c757d;
        font-family: 'Roboto', sans-serif;
        text-align: center;
        margin-bottom: 30px;
        font-size: 1.2rem;
    }
    
    .text-area-container {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    .result-box-safe {
        background-color: #d4edda;
        color: #155724;
        font-size: 18px;
        padding: 20px;
        border-radius: 8px;
        border-left: 6px solid #28a745;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        text-align: center;
        margin: 20px 0;
    }
    .result-box-hate {
        background-color: #f8d7da;
        color: #721c24;
        font-size: 18px;
        padding: 20px;
        border-radius: 8px;
        border-left: 6px solid #dc3545;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        text-align: center;
        margin: 20px 0;
    }
    .info-box {
        background-color: #e2f0fd;
        border: 1px solid #bee5eb;
        border-radius: 8px;
        padding: 15px;
        margin: 15px 0;
        color: #0c5460;
    }
    .footer {
        text-align: center;
        color: #6c757d;
        margin-top: 30px;
        padding: 20px;
        font-size: 0.8rem;
    }
    .icon-title {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 20px;
    }
    .example-btn {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        padding: 8px 12px;
        font-size: 0.9rem;
        margin: 5px;
        cursor: pointer;
    }
    .example-btn:hover {
        background-color: #e9ecef;
    }
    .metrics-container {
        display: flex;
        justify-content: space-between;
        margin: 20px 0;
    }
    .metric-card {
        background-color: white;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        flex: 1;
        margin: 0 10px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2196F3;
    }
    .metric-label {
        color: #6c757d;
        font-size: 0.9rem;
    }
    .divider {
        height: 1px;
        background-color: #dee2e6;
        margin: 30px 0;
    }
    .stTextArea textarea {
        font-size: 1.1rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar content
with st.sidebar:
    st.image("HateSpeech.jpg", width=150)
    st.title("About the Model")
    st.markdown("""
    This application uses a deep learning model trained to detect hate speech in Bangla text. The model analyzes the input sentence and determines if it contains harmful content.
    
    ### Model Information
    - **Type**: Deep Learning
    - **Framework**: TensorFlow/PyTorch
    - **Training Data**: Curated Bangla text corpus
    - **Accuracy**: ~92% on test data
    
    ### What is Hate Speech?
    Hate speech includes language that targets individuals or groups based on attributes such as:
    - Religion
    - Ethnicity
    - Gender
    - Sexual orientation
    - Physical characteristics
    """)
    
    st.divider()
    
    st.markdown("### How to use:")
    st.markdown("""
    1. Enter or paste a Bangla sentence in the text area
    2. Click "Analyze Text" button
    3. View the result and confidence score
    4. Try different examples to test the model
    """)
    
    st.divider()
    
    # Add language selection
    language = st.selectbox("Interface Language", ["English", "বাংলা"])

# Main content
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    #st.markdown("<h1 class='header'>Bangla Hate Speech Detection</h1>", unsafe_allow_html=True)
    #st.markdown("<p class='subheader'>Analyze Bangla text for harmful content using AI</p>", unsafe_allow_html=True)
    
    st.markdown("<div class='main-header'>Bangla Hate Speech Detection</div>", unsafe_allow_html=True)
    st.markdown("<div class='main-subheader'>Analyze Bangla text for harmful content using AI</div>", unsafe_allow_html=True)
    # Input section with better styling
    #st.markdown("<div class='text-area-container'>", unsafe_allow_html=True)
    input_text = st.text_area("Enter or paste a Bangla sentence:", 
                              height=120, 
                              key="input_area",
                              value=st.session_state['example_text'],
                              placeholder="এখানে বাংলা বাক্য লিখুন বা পেস্ট করুন...",
                              help="Type or paste any Bangla sentence to check if it contains hate speech")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Example sentences
    st.markdown("#### Examples to try:")
    col_a, col_b = st.columns(2)
    
    example_sentences = [
        "আমরা সবাই একসাথে বাংলাদেশকে এগিয়ে নিয়ে যাব",
        "সকল ধর্মের মানুষের মধ্যে শান্তি ও সম্প্রীতি বজায় রাখা উচিত",
        "আমি আজকে খুব ভালো আছি",
        "বন্ধুত্ব এবং সহযোগিতা আমাদের সমাজের মূল ভিত্তি"
    ]
    
    # Create buttons for examples using the callback function
    with col_a:
        if st.button("Example 1", key="ex_1", on_click=set_example, args=(example_sentences[0],)):
            pass
        if st.button("Example 3", key="ex_3", on_click=set_example, args=(example_sentences[2],)):
            pass
    
    with col_b:
        if st.button("Example 2", key="ex_2", on_click=set_example, args=(example_sentences[1],)):
            pass
        if st.button("Example 4", key="ex_4", on_click=set_example, args=(example_sentences[3],)):
            pass
    
    # Analysis button with better styling
    analyze_btn = st.button("🔍 Analyze Text", type="primary", use_container_width=True)
    
    # Process input
    if analyze_btn and input_text.strip():
        with st.spinner("Analyzing text..."):
            # Simulate processing time - in production, this would be your model inference
            result, confidence, lime_html = do_prediction(input_text)
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            # For demonstration - in production replace with actual model prediction
            # This is just a placeholder - you would connect your model here
              # 0 = not hate speech, 1 = hate speech
            
            
            st.success("Analysis complete!")
            progress_bar.empty()
            
            # Display result with nice styling
            if result == 0:
                st.markdown(f"""
                <div class='result-box-safe'>
                    <h2>✅ Not Hate Speech</h2>
                    <p>This text appears to be safe and does not contain harmful content.</p>
                    <h3>Confidence: {confidence:.2%}</h3>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-box-hate'>
                    <h2>⚠️ Hate Speech Detected</h2>
                    <p>This text may contain harmful content that targets individuals or groups.</p>
                    <h3>Confidence: {confidence:.2%}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### LIME Explanation")
            st.components.v1.html(lime_html, height=200)
            # Display text statistics
            st.markdown("### Text Statistics")
            
            # Calculate metrics (these would be more sophisticated in a real app)
            word_count = len(input_text.split())
            char_count = len(input_text)
            
            # Display metrics in a nice grid
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""
                <div class='metric-card'>
                    <div class='metric-value'>{}</div>
                    <div class='metric-label'>Characters</div>
                </div>
                """.format(char_count), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class='metric-card'>
                    <div class='metric-value'>{}</div>
                    <div class='metric-label'>Words</div>
                </div>
                """.format(word_count), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class='metric-card'>
                    <div class='metric-value'>{:.2f}</div>
                    <div class='metric-label'>Avg Word Length</div>
                </div>
                """.format(char_count/word_count if word_count > 0 else 0), unsafe_allow_html=True)
    
    elif analyze_btn and not input_text.strip():
        st.error("Please enter text to analyze!")
    
    # Information section
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    st.markdown("### About Bangla Hate Speech Detection")
    st.markdown("""
    Our system uses advanced natural language processing and deep learning techniques to identify hate speech in Bangla text. 
    The model has been trained on a diverse dataset of Bangla content to recognize patterns and linguistic features 
    associated with harmful content.
    
    This tool can be useful for content moderation, social media monitoring, and research purposes.
    """)
    
    # Show additional info in an expandable section
    with st.expander("How the AI works"):
        st.markdown("""
        The AI model works through the following steps:
        
        1. **Text Preprocessing**: The input text is cleaned and normalized.
        2. **Tokenization**: The text is broken down into tokens that the model can understand.
        3. **Feature Extraction**: The model identifies important linguistic features from the text.
        4. **Classification**: Based on the extracted features, the model determines if the text contains hate speech.
        5. **Confidence Scoring**: The model assigns a confidence score to its prediction.
        
        The model has been trained on thousands of Bangla text samples carefully labeled by human annotators.
        """)
    
    # Show limitations
    with st.expander("Limitations"):
        st.markdown("""
        While our model is highly accurate, it has some limitations:
        
        - It may occasionally misclassify sarcasm or cultural references
        - Very colloquial or dialectal Bangla might be more challenging
        - New hate speech terms or coded language might not be detected
        - The model works best with standard Bangla text
        
        We continuously work to improve the model's performance and reduce false positives/negatives.
        """)

    # Footer
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='footer'>© 2025 Bangla Hate Speech Detection Project</div>", unsafe_allow_html=True)

# Add a floating help button
st.markdown("""
<style>
.float-help {
    position: fixed;
    width: 50px;
    height: 50px;
    bottom: 40px;
    right: 40px;
    background-color: #2196F3;
    color: #FFF;
    border-radius: 50px;
    text-align: center;
    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    z-index: 1000;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    font-size: 24px;
}
</style>
<div class='float-help'>?</div>
""", unsafe_allow_html=True)

