import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

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

def make_positionless_heatmap(csv_path, output_img='positionless_digit_heatmap.png'):
    df = pd.read_csv(csv_path)
    
    counts = np.zeros((10, 1))
    
    for weight in df['weight']:
        w_str = f"{float(weight):.3f}"
        for c in w_str:
            if c.isdigit():
                counts[int(c),0] += 1
            
    # Convert to dataframe for better plotting labels
    heatmap_df = pd.DataFrame(
        counts, 
        index=[str(i) for i in range(10)],
        columns=['count']
    )
    
    plt.figure(figsize=(6, 8))
    sns.heatmap(heatmap_df, annot=True, fmt='g', cmap='YlGnBu')
    plt.title('Digit Frequencey Distribution (Positionless)')
    plt.xlabel('Total count')
    plt.ylabel('Digit value')
    plt.savefig(output_img)
    print(f"Heatmap saved to {output_img}")

if __name__ == "__main__":
    while True:
        clear_screen()
        print("Which of the following would you like to do?")
        print("1. Create a position-specific heatmap")
        print("2. Create a position agnostic heatmap")
        print("3. Create both heatmaps\n")

        choice = input("Select an option: ").strip()

        if choice == '1':
            make_position_heatmap('data/all_data.csv')
            break
        elif choice == '2':
            make_positionless_heatmap('data/all_data.csv')
            break
        elif choice == '3':
            make_position_heatmap('data/all_data.csv')
            make_positionless_heatmap('data/all_data.csv')
            break
        else:
            print("Invalid choice. Please try again.")
            input("Press Enter to continue...")
            continue