import numpy as np
import matplotlib.pyplot as plt
import sys

def visualize_mel_spectrogram(npy_path, save_path=None):
    """Visualize mel-spectrogram from .npy file"""

    # Load mel-spectrogram
    mel = np.load(npy_path)

    # Create figure
    plt.figure(figsize=(12, 4))
    plt.imshow(mel.T, aspect='auto', origin='lower', interpolation='none', cmap='viridis')
    plt.colorbar(format='%+2.0f')
    plt.title(f'Mel-Spectrogram: {npy_path}')
    plt.xlabel('Time Frame')
    plt.ylabel('Mel Frequency Bin')
    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved mel-spectrogram visualization to: {save_path}")
    else:
        plt.show()

    plt.close()

    # Print statistics
    print(f"\nMel-spectrogram shape: {mel.shape}")
    print(f"Time frames: {mel.shape[0]}, Mel bins: {mel.shape[1]}")
    print(f"Value range: [{mel.min():.3f}, {mel.max():.3f}]")
    print(f"Mean: {mel.mean():.3f}, Std: {mel.std():.3f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_mel.py <path_to_npy_file> [output_png_path]")
        print("\nExample:")
        print("  python visualize_mel.py logdir-tacotron2/test_samples/step_100000/2025-11-22_16-42-19.npy")
        print("  python visualize_mel.py logdir-tacotron2/test_samples/step_100000/2025-11-22_16-42-19.npy mel_viz.png")
        sys.exit(1)

    npy_path = sys.argv[1]
    save_path = sys.argv[2] if len(sys.argv) > 2 else None

    visualize_mel_spectrogram(npy_path, save_path)
