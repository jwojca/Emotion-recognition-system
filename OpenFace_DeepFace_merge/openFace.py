from collections import Counter

def GetDominantEmotion(data):
    """
    Returns the dominant emotion and its percentage in a given array of emotions.

    Args:
        data: A NumPy array of emotions.

    Returns:
        A tuple containing the dominant emotion (string) and its percentage (float).
    """
    counts = Counter(data)
    dominantEmotion = counts.most_common(1)[0][0]
    dominantPercentage = counts[dominantEmotion] / len(data) * 100
    return (dominantEmotion, dominantPercentage)