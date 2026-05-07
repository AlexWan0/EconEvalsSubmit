def included_samples(
        result_hyp: dict[str, dict]    
    ) -> list[str]:
    return [
        k
        for k, v in result_hyp['sample2score'].items()
        if v > 0.6
    ]

def simplify_output(
        result: dict[str, dict]
    ) -> list[tuple[str, float, list[str]]]:

    return sorted([
        (k, float(v['diff_w_significance']['mu']), included_samples(v))
        for k, v in result.items()
    ], key=lambda x: x[1], reverse=True)
