import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def make_position_heatmap(csv_path, output_img='digit_heatmap.png'):
    df = pd.read_csv(csv_path)
    
    # create a 10 row (digits 0-9) by 4 column (positions) matrix
    counts = np.zeros((10, 4))
    
    for weight in df['weight']:
        w_str = f"{float(weight):.3f}"
        
        # skip the decimal point
        digits = [w_str[0], w_str[2], w_str[3], w_str[4]]
        
        for pos, digit in enumerate(digits):
            counts[int(digit), pos] += 1
            
    # Convert to dataframe for better plotting labels
    heatmap_df = pd.DataFrame(
        counts, 
        index=[str(i) for i in range(10)],
        columns=['ones', 'tenths', 'hundreths', 'thousandths']
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_df, annot=True, fmt='g', cmap='YlGnBu')
    plt.title('Digit Position Distribution')
    plt.xlabel('Position in X.XXX')
    plt.ylabel('Digit Value')
    plt.savefig(output_img)
    print(f"Heatmap saved to {output_img}")

if __name__ == "__main__":
    make_position_heatmap('data/all_data.csv')