import pandas as pd
import pickle as pkl
import streamlit as st
from PIL import Image as img

# Load the trained model
pickle_in = open('classifier.pkl', 'rb')
loaded_classifier = pkl.load(pickle_in)

column_names = [
        'length_url', 'length_hostname', 'ip', 'nb_dots', 'nb_hyphens', 'nb_at', 'nb_qm', 'nb_and', 'nb_or', 'nb_eq',
        'nb_underscore', 'nb_tilde', 'nb_percent', 'nb_slash', 'nb_star', 'nb_colon', 'nb_comma', 'nb_semicolon',
        'nb_dollar', 'nb_space', 'nb_www', 'nb_com', 'nb_dslash', 'http_in_path', 'https_token', 'ratio_digits_url',
        'ratio_digits_host', 'punycode', 'port', 'tld_in_path', 'tld_in_subdomain', 'abnormal_subdomain', 'nb_subdomains',
        'prefix_suffix', 'random_domain', 'shortening_service', 'path_extension', 'nb_redirection', 'nb_external_redirection',
        'length_words_raw', 'char_repeat', 'shortest_words_raw', 'shortest_word_host', 'shortest_word_path',
        'longest_words_raw', 'longest_word_host', 'longest_word_path', 'avg_words_raw', 'avg_word_host', 'avg_word_path',
        'phish_hints', 'domain_in_brand', 'brand_in_subdomain', 'brand_in_path', 'suspecious_tld', 'statistical_report',
        'nb_hyperlinks', 'ratio_intHyperlinks', 'ratio_extHyperlinks', 'ratio_nullHyperlinks', 'nb_extCSS',
        'ratio_intRedirection', 'ratio_extRedirection', 'ratio_intErrors', 'ratio_extErrors', 'login_form', 'external_favicon',
        'links_in_tags', 'submit_email', 'ratio_intMedia', 'ratio_extMedia', 'sfh', 'iframe', 'popup_window', 'safe_anchor',
        'onmouseover', 'right_clic', 'empty_title', 'domain_in_title', 'domain_with_copyright', 'whois_registered_domain',
        'domain_registration_length', 'domain_age', 'web_traffic', 'dns_record', 'google_index', 'page_rank'
    ]

# Define a function for making predictions
def prediction1(inputs):
    input_values = []

    
    for column in column_names:
        try:
            input_value = float(inputs[column])
            input_values.append(input_value)
        except ValueError:
            return "Invalid input. Please enter a valid number for {}".format(column)
    prediction = loaded_classifier.predict([input_values])
    return prediction

# Main function for defining the web page
def main():
    st.markdown('<div style="background-color:#FFF00;padding:16px"><h1 style="color:#000000;text-align:center;">PHISHING WEBSITES DETECTION</h1></div>', unsafe_allow_html=True)
    
    user_inputs = {}


    for column_name in column_names:
        user_input = st.text_input(f"Enter {column_name}")
        user_inputs[column_name] = user_input

    result = ""

    # Add a "Predict" button to trigger predictions
    if st.button("Predict"):
        result = prediction1(user_inputs)
    
    # Display the prediction result
    if (result==0):
        st.success(f'The website is NOT fraud')
    elif (result==1):
        st.success(f'The website is fraud')
    else:
        st.success(f'MODEL IN PROCESS')



if __name__ == '__main__':
    main()
