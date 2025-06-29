import numpy as np
import matplotlib.pyplot as plt
from mnist_neural_network import NeuralNetwork, load_and_preprocess_mnist, visualize_predictions
import time

def experiment_with_architectures():
    """Experiment with different network architectures"""
    print("🔬 Experimenting with Different Architectures")
    print("=" * 50)
    
    # Load data (smaller subset for faster experimentation)
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_mnist()
    
    # Use smaller dataset for quick experiments
    X_train_small = X_train[:5000]
    y_train_small = y_train[:5000]
    X_val_small = X_val[:1000]
    y_val_small = y_val[:1000]
    
    architectures = [
        ([64], "Single Hidden Layer (64)"),
        ([128], "Single Hidden Layer (128)"),
        ([128, 64], "Two Hidden Layers (128, 64)"),
        ([256, 128, 64], "Three Hidden Layers (256, 128, 64)"),
        ([512, 256], "Deep Wide (512, 256)")
    ]
    
    results = []
    
    for hidden_sizes, name in architectures:
        print(f"\nTesting: {name}")
        
        # Create and train network
        nn = NeuralNetwork(
            input_size=784,
            hidden_sizes=hidden_sizes,
            output_size=10,
            learning_rate=0.01
        )
        
        start_time = time.time()
        nn.train(X_train_small, y_train_small, X_val_small, y_val_small, 
                epochs=30, batch_size=64, verbose=False)
        training_time = time.time() - start_time
        
        # Evaluate
        val_acc = nn.calculate_accuracy(X_val_small, y_val_small)
        
        results.append({
            'name': name,
            'architecture': hidden_sizes,
            'accuracy': val_acc,
            'time': training_time,
            'parameters': sum(w.size for w in nn.weights) + sum(b.size for b in nn.biases)
        })
        
        print(f"Validation Accuracy: {val_acc:.4f}, Time: {training_time:.2f}s")
    
    # Plot comparison
    names = [r['name'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    times = [r['time'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy comparison
    bars1 = ax1.bar(range(len(names)), accuracies, color='skyblue')
    ax1.set_xlabel('Architecture')
    ax1.set_ylabel('Validation Accuracy')
    ax1.set_title('Architecture Comparison - Accuracy')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # Training time comparison
    bars2 = ax2.bar(range(len(names)), times, color='lightcoral')
    ax2.set_xlabel('Architecture')
    ax2.set_ylabel('Training Time (seconds)')
    ax2.set_title('Architecture Comparison - Training Time')
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, time_val in zip(bars2, times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{time_val:.1f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\n📊 Architecture Comparison Summary:")
    print("-" * 70)
    for result in results:
        print(f"{result['name']:25} | Acc: {result['accuracy']:.4f} | "
              f"Time: {result['time']:5.1f}s | Params: {result['parameters']:,}")
    
    return results


def experiment_with_learning_rates():
    """Experiment with different learning rates"""
    print("\n🎯 Experimenting with Learning Rates")
    print("=" * 50)
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_mnist()
    
    # Use smaller dataset
    X_train_small = X_train[:5000]
    y_train_small = y_train[:5000]
    X_val_small = X_val[:1000]
    y_val_small = y_val[:1000]
    
    learning_rates = [0.001, 0.01, 0.05, 0.1, 0.2]
    results = []
    
    plt.figure(figsize=(15, 10))
    
    for i, lr in enumerate(learning_rates):
        print(f"\nTesting Learning Rate: {lr}")
        
        nn = NeuralNetwork(
            input_size=784,
            hidden_sizes=[128, 64],
            output_size=10,
            learning_rate=lr
        )
        
        nn.train(X_train_small, y_train_small, X_val_small, y_val_small,
                epochs=50, batch_size=64, verbose=False)
        
        final_acc = nn.val_accuracies[-1] if nn.val_accuracies else nn.train_accuracies[-1]
        results.append({'lr': lr, 'accuracy': final_acc, 'history': nn.val_accuracies or nn.train_accuracies})
        
        # Plot training curves
        plt.subplot(2, 3, i+1)
        plt.plot(nn.train_accuracies, label='Train', alpha=0.7)
        if nn.val_accuracies:
            plt.plot(nn.val_accuracies, label='Validation', alpha=0.7)
        plt.title(f'LR = {lr} (Final Acc: {final_acc:.3f})')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        print(f"Final Accuracy: {final_acc:.4f}")
    
    # Summary plot
    plt.subplot(2, 3, 6)
    lrs = [r['lr'] for r in results]
    accs = [r['accuracy'] for r in results]
    plt.semilogx(lrs, accs, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Learning Rate')
    plt.ylabel('Final Validation Accuracy')
    plt.title('Learning Rate vs Final Accuracy')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Find best learning rate
    best_result = max(results, key=lambda x: x['accuracy'])
    print(f"\n🏆 Best Learning Rate: {best_result['lr']} (Accuracy: {best_result['accuracy']:.4f})")
    
    return results


def analyze_training_dynamics():
    """Analyze training dynamics and convergence"""
    print("\n📈 Analyzing Training Dynamics")
    print("=" * 50)
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_mnist()
    
    # Use moderate dataset size
    X_train_med = X_train[:10000]
    y_train_med = y_train[:10000]
    X_val_med = X_val[:2000]
    y_val_med = y_val[:2000]
    
    # Train with detailed tracking
    nn = NeuralNetwork(
        input_size=784,
        hidden_sizes=[128, 64],
        output_size=10,
        learning_rate=0.01
    )
    
    print("Training with detailed monitoring...")
    nn.train(X_train_med, y_train_med, X_val_med, y_val_med,
            epochs=100, batch_size=64, verbose=True)
    
    # Analyze convergence
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(nn.train_losses, label='Training Loss', alpha=0.8)
    axes[0, 0].plot(nn.val_losses, label='Validation Loss', alpha=0.8)
    axes[0, 0].set_title('Loss Convergence')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Cross-Entropy Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[0, 1].plot(nn.train_accuracies, label='Training Accuracy', alpha=0.8)
    axes[0, 1].plot(nn.val_accuracies, label='Validation Accuracy', alpha=0.8)
    axes[0, 1].set_title('Accuracy Convergence')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Overfitting analysis
    train_val_gap = np.array(nn.train_accuracies) - np.array(nn.val_accuracies)
    axes[1, 0].plot(train_val_gap, color='red', alpha=0.8)
    axes[1, 0].set_title('Overfitting Analysis (Train - Val Accuracy)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy Gap')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Learning rate analysis (loss gradient)
    loss_gradient = np.gradient(nn.train_losses)
    axes[1, 1].plot(loss_gradient, color='purple', alpha=0.8)
    axes[1, 1].set_title('Loss Gradient (Learning Speed)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss Gradient')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    # Print analysis
    final_gap = train_val_gap[-1]
    max_gap = np.max(train_val_gap)
    convergence_epoch = np.argmin(np.abs(loss_gradient[10:])) + 10  # Skip first 10 epochs
    
    print(f"\n📊 Training Analysis:")
    print(f"Final Train-Val Gap: {final_gap:.4f}")
    print(f"Maximum Overfitting: {max_gap:.4f}")
    print(f"Convergence around epoch: {convergence_epoch}")
    print(f"Final Training Accuracy: {nn.train_accuracies[-1]:.4f}")
    print(f"Final Validation Accuracy: {nn.val_accuracies[-1]:.4f}")
    
    return nn


def test_robustness():
    """Test model robustness with noisy data"""
    print("\n🛡️ Testing Model Robustness")
    print("=" * 50)
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_mnist()
    
    # Train a model
    nn = NeuralNetwork(input_size=784, hidden_sizes=[128, 64], output_size=10, learning_rate=0.01)
    nn.train(X_train[:10000], y_train[:10000], epochs=50, batch_size=64, verbose=False)
    
    # Test with different noise levels
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    accuracies = []
    
    test_subset = X_test[:1000]
    test_labels = y_test[:1000]
    
    for noise_level in noise_levels:
        # Add noise to test data
        noisy_data = test_subset + np.random.normal(0, noise_level, test_subset.shape)
        noisy_data = np.clip(noisy_data, 0, 1)  # Keep in valid range
        
        # Test accuracy
        accuracy = nn.calculate_accuracy(noisy_data, test_labels)
        accuracies.append(accuracy)
        
        print(f"Noise Level {noise_level:.1f}: Accuracy = {accuracy:.4f}")
    
    # Plot robustness
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, accuracies, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Noise Level (Standard Deviation)')
    plt.ylabel('Accuracy')
    plt.title('Model Robustness to Input Noise')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    
    # Add value labels
    for i, (noise, acc) in enumerate(zip(noise_levels, accuracies)):
        plt.annotate(f'{acc:.3f}', (noise, acc), textcoords="offset points", 
                    xytext=(0,10), ha='center')
    
    plt.show()
    
    # Show examples of noisy images
    fig, axes = plt.subplots(2, 6, figsize=(15, 6))
    
    sample_idx = 0
    original_image = test_subset[sample_idx].reshape(28, 28)
    
    for i, noise_level in enumerate(noise_levels):
        # Original
        axes[0, i].imshow(original_image, cmap='gray')
        axes[0, i].set_title(f'Noise: {noise_level:.1f}')
        axes[0, i].axis('off')
        
        # Noisy version
        noisy_image = original_image + np.random.normal(0, noise_level, original_image.shape)
        noisy_image = np.clip(noisy_image, 0, 1)
        axes[1, i].imshow(noisy_image, cmap='gray')
        axes[1, i].axis('off')
        
        # Predict
        noisy_flat = noisy_image.reshape(1, -1)
        pred = nn.predict(noisy_flat)[0]
        true_label = test_labels[sample_idx]
        axes[1, i].set_title(f'Pred: {pred} (True: {true_label})', 
                           color='green' if pred == true_label else 'red')
    
    plt.suptitle(f'Robustness Test - Original Digit: {test_labels[sample_idx]}')
    plt.tight_layout()
    plt.show()
    
    return accuracies


def main():
    """Run all experiments"""
    print("🚀 Advanced Neural Network Experiments")
    print("=" * 60)
    
    # Run experiments
    print("\n1. Architecture Experiments")
    arch_results = experiment_with_architectures()
    
    print("\n2. Learning Rate Experiments")
    lr_results = experiment_with_learning_rates()
    
    print("\n3. Training Dynamics Analysis")
    trained_model = analyze_training_dynamics()
    
    print("\n4. Robustness Testing")
    robustness_results = test_robustness()
    
    print("\n🎉 All experiments completed!")
    print("\n📋 Summary:")
    print("- Tested different architectures")
    print("- Optimized learning rates")
    print("- Analyzed training dynamics")
    print("- Evaluated model robustness")
    
    return {
        'architectures': arch_results,
        'learning_rates': lr_results,
        'trained_model': trained_model,
        'robustness': robustness_results
    }


if __name__ == "__main__":
    results = main() 