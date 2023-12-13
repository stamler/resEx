import os
from pypdf import PdfReader
import pandas as pd
import requests

from typing import List

def extract_text_from_pdf(pdf_path: str) -> str:
  """
  Extracts text from a PDF file.

  Args:
    pdf_path (str): The path to the PDF file.

  Returns:
    str: The extracted text from the PDF file.
  """
  with open(pdf_path, 'rb') as file:
    pdf_reader = PdfReader(file)
    # Using a list comprehension to iterate over each page in the PDF file and
    # extract the text, joining them with newline characters and returning the
    # result as a string
    return '\n'.join([page.extract_text() for page in pdf_reader.pages])


def load_data(folder_path: str) -> List[tuple]:
  """
  Iterates over a folder and extracts text from all PDF files within the folder.

  Args:
    folder_path (str): The path to the folder.

  Returns:
    List[tuple]: A list of tuples containing the filename and extracted text from all PDF files within the folder.
  """
  results = []
  for filename in os.listdir(folder_path):
    if filename.endswith('.pdf'):
      pdf_path = os.path.join(folder_path, filename)
      text = extract_text_from_pdf(pdf_path)
      results.append((filename, text))
  return results


def query_openai_api(instructions: str, resume_text: str) -> dict:
  """
  Queries the OpenAI API with instructions and resume text.

  Args:
    instructions (str): The instructions for the OpenAI API.
    resume_text (str): The text extracted from the resume.

  Returns:
    dict: The extracted information from the resume as a dictionary.
  """
  # Make a request to the OpenAI API with instructions and resume text
  response = requests.post('https://api.openai.com/v1/extract', json={'instructions': instructions, 'resume_text': resume_text})
  # Parse the JSON response into a dictionary
  extracted_info = response.json()
  return extracted_info


def save_dataframe_as_csv(dataframe: pd.DataFrame, csv_path: str):
  """
  Saves a pandas DataFrame as a CSV file.

  Args:
    dataframe (pd.DataFrame): The DataFrame to be saved.
    csv_path (str): The path to save the CSV file.
  """
  dataframe.to_csv(csv_path, index=False)


# Read instructions from prompt.txt
with open('prompt.txt', 'r') as file:
  instructions = file.read()

folder_path: str = './resumes'
resumes = load_data(folder_path)

# Create an empty DataFrame
data = pd.DataFrame(columns=['Filename', 'Extracted Text'])

# Iterate over each resume and query the OpenAI API
for filename, extracted_text in resumes:
  # print the filename to the console to track progress
  print(filename)
  extracted_info = query_openai_api(instructions, extracted_text)
  # Create a dictionary to store the extracted information
  info_dict = {'Filename': filename, 'Extracted Text': extracted_text}
  # Add the extracted information to the dictionary
  info_dict.update(extracted_info)
  # Append the dictionary to the DataFrame
  data = pd.concat([data, pd.DataFrame(info_dict, index=[0])], ignore_index=True)

# Save the DataFrame as a CSV file
save_dataframe_as_csv(data, 'output.csv')


