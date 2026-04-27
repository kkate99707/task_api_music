import jiwer

def evaluate_text(reference, hypothesis):
    transformation = jiwer.compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.Strip(),
        jiwer.RemoveMultipleSpaces()
    ])
    
    ref_clean = transformation(reference)
    hyp_clean = transformation(hypothesis)

    wer = jiwer.wer(ref_clean, hyp_clean)

    accuracy = max(0, 1 - wer)
    
    return {
        "wer": round(wer, 4),
        "accuracy": round(accuracy * 100, 2)
    }
