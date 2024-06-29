import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


def update_res_column(df):
    opposite_mapping = {'low': 'high', 'high': 'low'}
    df['predicted'] = df.apply(lambda row: row['attachment'] if row['correct'] == row['output'] else opposite_mapping[row['attachment']], axis=1)
    return df

def compute(df, file_name):
    label_mapping = {'low': 0, 'high': 1}
    df['attachment'] = df['attachment'].map(label_mapping)
    df['predicted'] = df['predicted'].map(label_mapping)

    y_true = df['attachment']
    y_pred = df['predicted']

    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro')

    precision_micro = precision_score(y_true, y_pred, average='micro')
    recall_micro = recall_score(y_true, y_pred, average='micro')
    f1_micro = f1_score(y_true, y_pred, average='micro')

    precision_weighted = precision_score(y_true, y_pred, average='weighted')
    recall_weighted = recall_score(y_true, y_pred, average='weighted')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    metrics = {
    'Accuracy': accuracy,
    'Precision (Macro)': precision_macro,
    'Recall (Macro)': recall_macro,
    'F1 Score (Macro)': f1_macro,
    'Precision (Micro)': precision_micro,
    'Recall (Micro)':   recall_macro,
    'F1 Score (Micro)': f1_micro,
    'Precision (Weighted)': precision_weighted,
    'Recall (Weighted)': recall_weighted,
    'F1 Score (Weighted)': f1_weighted
    }
    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Score'])
    metrics_df.to_csv(f'res/{file_name}.csv', index=False)


def plot(file_groups, group_names, selected_metrics):
    plt.figure(figsize=(12, 8))
    
    for group, group_name in zip(file_groups, group_names):
        all_dfs = []
        
        for file_path in group:
            df = pd.read_csv(file_path)
            all_dfs.append(df)
        
        combined_df = pd.concat(all_dfs).groupby('Metric', as_index=False).mean()
        
        filtered_df = combined_df[combined_df['Metric'].isin(selected_metrics)]
        
        plt.plot(filtered_df['Metric'], filtered_df['Score'], marker='o', label=group_name)
    
    plt.xlabel('Metric')
    plt.ylabel('Score')
    #plt.title('Performance Metrics')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.show()

# files in the res folder
file_groups = [
    ['res/gpt-3.5-turbo_1.csv', 'res/gpt-3.5-turbo_2.csv'],
    ['res/gpt-4o_1.csv', 'res/gpt-4o_2.csv'],
    ['res/llama3_1.csv', 'res/llama3_2.csv']
]

group_names = ['gpt-3.5-turbo', 'gpt-4o', 'llama3']
selected_metrics = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1 Score (Macro)']

#plot(file_groups, group_names, selected_metrics)



def plot1(paths):
    new_dataframes = [pd.read_csv(file_path) for file_path in paths]

    gpt_35_turbo_new_df = new_dataframes[0]
    gpt_4o_new_df = new_dataframes[1]
    llama3_new_df = new_dataframes[2]
    claude_new_df = new_dataframes[3]
    gem_new_df = new_dataframes[4]

    gpt_35_turbo_predicted_new_counts = gpt_35_turbo_new_df['predicted'].value_counts()
    gpt_4o_predicted_new_counts = gpt_4o_new_df['predicted'].value_counts()
    llama3_predicted_new_counts = llama3_new_df['predicted'].value_counts()
    claude_predicted_new_counts = claude_new_df['predicted'].value_counts()
    gem_predicted_new_counts = gem_new_df['predicted'].value_counts()

    plt.figure(figsize=(12, 6))

    plt.plot(gpt_35_turbo_predicted_new_counts.index, gpt_35_turbo_predicted_new_counts.values, marker='o', linestyle='-', color='green', label='GPT-3.5-Turbo')
    plt.plot(gpt_4o_predicted_new_counts.index, gpt_4o_predicted_new_counts.values, marker='o', linestyle='-', color='red', label='GPT-4o')
    plt.plot(llama3_predicted_new_counts.index, llama3_predicted_new_counts.values, marker='o', linestyle='-', color='purple', label='Llama3')
    plt.plot(claude_predicted_new_counts.index, claude_predicted_new_counts.values, marker='o', linestyle='-', color='blue', label='Claude-3-Opus')
    plt.plot(gem_predicted_new_counts.index, gem_predicted_new_counts.values, marker='*', linestyle='dotted', color='yellow', label='Genimi-Pro')

    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.savefig('plot_p2.png')
    plt.savefig('plot_p2.pdf')
    plt.show()


# Load the new uploaded files
paths = [
    'gpt-3.5-turbo_2.csv',
    'gpt-4o_2.csv',
    'llama3_2.csv',
    'claude-3-opus_2.csv',
    'gemini-pro_2.csv'
]

plot1(paths)

# if __name__ == '__main__':
#     file = 'claude-3-opus-20240229_1.xlsx'
#     df = pd.read_excel(file)
#     df = update_res_column(df)
#     df.to_csv(f'{os.path.splitext(os.path.basename(file))[0]}.csv', index=False)
    #compute(df=df, file_name=os.path.splitext(os.path.basename(file))[0])

