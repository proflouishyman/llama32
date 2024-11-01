# Date: 2024-11-01
# Purpose: Clean up text by applying specific rules to remove unwanted markers and characters

import re

# Original text
text = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

<|image|>EXTRACT TEXT EXACTLY AND COMPLETELY AS WRITTEN. Preserve indents and whitespaces. Do not provide summaries, interpretations, or any additional commentary. Do NOT add additional characters, like asterisks, to format the information.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Mr. P. E. Gilmore, Organizer,

Rural Free Delivery,

Fort Smith, Arkansas.

Dear Sir and Brother:

Your favor of the 21st instant has been duly received, and I have read with interest the report you make as'to the legislative work in the State of Arkansas. I would like you to write me, from time to time, as to any matters of interest affecting the wage workers of Fort Smith, of the vicinity and of the entire state.

Enclosed you will please find copy of list of affiliated organizations, giving the names and addresses of their respective secretaries, from which you will note that the President of the Amalgamated Association of Street and Electric Railway Employees is Mr. W. I. Meron, 45 Hodges Block, Detroit, Michigan.

With kind regards and hoping to hear from you whenever convenient, I am,

Fraternally yours,

Samuel Gompers,

President

American Federation of Labor.

Per R.J.G.<|eot_id|>"""

# Clean up the text
cleaned_text = re.sub(r'<\|begin_of_text\|>', '', text)  # Remove <|begin_of_text|>
cleaned_text = re.sub(r'<\|start_header_id\|>.*?<\|end_header_id\|>', '', cleaned_text, flags=re.DOTALL)  # Remove <|start_header_id|> to <|end_header_id|>
cleaned_text = re.sub(r'<\|image\|>.*?<\|eot_id\|>', '', cleaned_text, flags=re.DOTALL)  # Remove <|image|> to <|eot_id|>
cleaned_text = re.sub(r'<\|eot_id\|>', '', cleaned_text)  # Remove <|eot_id|>
cleaned_text = re.sub(r'\*\*', '', cleaned_text)  # Remove double asterisks

# Print the cleaned text
print(cleaned_text.strip())
