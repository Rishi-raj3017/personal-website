#!/usr/bin/env python3
"""
Simple example of using the Neural Network from scratch
This is a minimal example to get you started quickly
"""

import numpy as np
import matplotlib.pyplot as plt
from mnist_neural_network import NeuralNetwork, load_and_preprocess_mnist

def quick_demo():
    """Quick demonstration of the neural network"""
    print("🚀 Quick Neural Network Demo")
    print("=" * 40)
    
    # Load MNIST data
    print("Loading MNIST dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_mnist()
    
    # Use smaller dataset for quick demo
    print("Using subset of data for quick demo...")
    X_train_small = X_train[:5000]  # 5000 training samples
    y_train_small = y_train[:5000]
    X_val_small = X_val[:1000]      # 1000 validation samples
    y_val_small = y_val[:1000]
    X_test_small = X_test[:500]     # 500 test samples
    y_test_small = y_test[:500]
    
    print(f"Training set: {X_train_small.shape}")
    print(f"Validation set: {X_val_small.shape}")
    print(f"Test set: {X_test_small.shape}")
    
    # Create neural network
    print("\nCreating neural network...")
    nn = NeuralNetwork(
        input_size=784,      # 28x28 pixels
        hidden_sizes=[128, 64],  # Two hidden layers
        output_size=10,      # 10 digit classes
        learning_rate=0.01
    )
    
    print("Network architecture: 784 → 128 → 64 → 10")
    total_params = sum(w.size for w in nn.weights) + sum(b.size for b in nn.biases)
    print(f"Total parameters: {total_params:,}")
    
    # Train the network
    print("\nTraining neural network...")
    nn.train(
        X_train_small, y_train_small,
        X_val_small, y_val_small,
        epochs=30,           # Quick training
        batch_size=64,
        verbose=True
    )
    
    # Evaluate performance
    print("\nEvaluating performance...")
    train_acc = nn.calculate_accuracy(X_train_small, y_train_small)
    val_acc = nn.calculate_accuracy(X_val_small, y_val_small)
    test_acc = nn.calculate_accuracy(X_test_small, y_test_small)
    
    print(f"Training Accuracy:   {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy:       {test_acc:.4f}")
    
    # Show some predictions
    print("\nMaking predictions on test samples...")
    sample_indices = np.random.choice(len(X_test_small), 6, replace=False)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()
    
    for i, idx in enumerate(sample_indices):
        # Get image and true label
        image = X_test_small[idx].reshape(28, 28)
        true_label = y_test_small[idx]
        
        # Make prediction
        prediction = nn.predict(X_test_small[idx:idx+1])[0]
        probabilities = nn.predict_proba(X_test_small[idx:idx+1])[0]
        confidence = np.max(probabilities)
        
        # Plot
        axes[i].imshow(image, cmap='gray')
        color = 'green' if prediction == true_label else 'red'
        axes[i].set_title(f'True: {true_label}, Pred: {prediction}\nConfidence: {confidence:.3f}', 
                         color=color)
        axes[i].axis('off')
    
    plt.suptitle('Sample Predictions (Green=Correct, Red=Wrong)')
    plt.tight_layout()
    plt.show()
    
    # Plot training history
    print("\nPlotting training history...")
    nn.plot_training_history()
    
    # Save the model
    print("\nSaving trained model...")
    nn.save_model('quick_demo_model.pkl')
    
    print("\n✅ Demo completed successfully!")
    print(f"Final test accuracy: {test_acc:.1%}")
    
    return nn

def load_and_test_saved_model():
    """Demonstrate loading a saved model"""
    print("\n🔄 Loading Saved Model Demo")
    print("=" * 40)
    
    # Create a new neural network instance
    nn_loaded = NeuralNetwork()
    
    try:
        # Load the saved model
        nn_loaded.load_model('quick_demo_model.pkl')
        
        # Load test data
        _, _, X_test, _, _, y_test = load_and_preprocess_mnist()
        X_test_small = X_test[:100]
        y_test_small = y_test[:100]
        
        # Test the loaded model
        accuracy = nn_loaded.calculate_accuracy(X_test_small, y_test_small)
        print(f"Loaded model accuracy: {accuracy:.4f}")
        
        # Make some predictions
        predictions = nn_loaded.predict(X_test_small[:5])
        print(f"Sample predictions: {predictions}")
        print(f"True labels:        {y_test_small[:5]}")
        
        print("✅ Model loading successful!")
        
    except FileNotFoundError:
        print("❌ No saved model found. Run the quick demo first!")
    
    return nn_loaded

if __name__ == "__main__":
    # Run quick demo
    trained_model = quick_demo()
    
    # Demonstrate model loading
    loaded_model = load_and_test_saved_model()
    
    print("\n🎉 All demos completed!")
    print("\nNext steps:")
    print("1. Try running 'python demo_advanced_features.py' for advanced experiments")
    print("2. Modify the network architecture in NeuralNetwork()")
    print("3. Experiment with different learning rates and epochs")
    print("4. Try the network on your own datasets!") 