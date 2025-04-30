from vis_utils import visualize_hot, visualize_hot_localization
from data_utils import process_examples_for_localization_training, process_examples_for_repeating_training, load_and_preprocess_dataset
import os, json

def visualize_data(vis_type='localization', num=10):
    """_summary_

    Args:
        vis_type: 'localization' or 'repeating'
        num (int): The number of examples to visualize.
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Question and Answer Highlights</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
            }
            .highlight {
                background-color: #FFFF00; /* Yellow background for visibility */
                font-weight: bold; /* Bold text for emphasis */
            }
            .container {
                display: flex;
                justify-content: space-between;
                margin-bottom: 20px;
            }
            .question, .answer {
                flex: 1;
                padding: 10px;
            }
        </style>
    </head>
    <body>
    """
    
    print("Loading dataset...")
    train_ds, val_ds, test_ds = load_and_preprocess_dataset(no_test_split=True)
    
    print(f"Dataset loaded: {len(train_ds)} train, {len(val_ds)} val test samples")
    
    if vis_type == 'localization':
        processed_ds = process_examples_for_localization_training(train_ds)
        for i in range(num):
            input = processed_ds[i]['input']
            output = processed_ds[i]['output']
            html = visualize_hot_localization(f"<question>{input}</question><answer>{output}</answer>")
            html_content += html
            
    elif vis_type == 'repeating':
        processed_ds = process_examples_for_repeating_training(train_ds)
        for i in range(num):
            input = processed_ds[i]['input']
            output = processed_ds[i]['output']
            html = visualize_hot(output)
            html_content += html
        
    # Close the HTML tags
    html_content += """
    </body>
    </html>
    """
    
    return html_content

def save_html(content, filename='visualization.html'):
    with open(filename, 'w') as f:
        f.write(content)
    print(f"HTML content saved to {filename}")

if __name__ == "__main__":
    # Example usage
    vis_type='repeating'
    
    html_content = visualize_data(vis_type=vis_type, num=10)
    save_folder = 'visualization'
    os.makedirs(save_folder, exist_ok=True)
    filename = os.path.join(save_folder, f'vis_{vis_type}.html')
    save_html(html_content, filename=filename)
    
    
    
    
    
        
    
    
    
    
    
    
    
    