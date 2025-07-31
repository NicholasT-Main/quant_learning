def grouping_any_correlation(minimum_correlation, correlation_matrix, min_group_size=2):
    """
    Groups stocks where each stock correlates with AT LEAST ONE other stock in the group.
    This is the original greedy approach - fastest but least restrictive.

    Complexity: O(n² × g) where n=stocks, g=avg group size
    """
    groups = []

    # For each stock, see if it fits into an existing group
    for stock in correlation_matrix.columns:
        placed = False

        # Check if this stock belongs to any existing group
        for group in groups:
            # If this stock correlates well with ANY stock already in the group
            if any(correlation_matrix.loc[stock, existing_stock] >= minimum_correlation
                   for existing_stock in group if existing_stock != stock):
                group.append(stock)
                placed = True
                break

        # If it doesn't fit anywhere, start a new group
        if not placed:
            groups.append([stock])

    # Clean up - only keep groups with minimum size
    final_groups = [group for group in groups if len(group) >= min_group_size]

    return final_groups


def grouping_all_correlate(minimum_correlation, correlation_matrix, min_group_size=2):
    """
    Groups stocks where EVERY stock correlates with EVERY other stock in the group.
    This creates tight clusters - most restrictive but highest quality.

    Complexity: O(n³) - must verify all pairwise correlations
    """
    groups = []

    for stock in correlation_matrix.columns:
        placed = False

        # Check if this stock can join any existing group
        for group in groups:
            # Check if this stock correlates with ALL stocks in the group
            if all(correlation_matrix.loc[stock, existing_stock] >= minimum_correlation
                   for existing_stock in group if existing_stock != stock):
                # Double check: make sure adding this stock maintains the all-correlate property
                # (This is technically redundant but ensures group integrity)
                group.append(stock)
                placed = True
                break

        # If it doesn't fit anywhere, start a new group
        if not placed:
            groups.append([stock])

    # Clean up and validate final groups
    final_groups = []
    for group in groups:
        if len(group) >= min_group_size:
            # Verify the group actually satisfies the all-correlate condition
            valid_group = True
            for i, stock1 in enumerate(group):
                for stock2 in group[i + 1:]:
                    if correlation_matrix.loc[stock1, stock2] < minimum_correlation:
                        valid_group = False
                        break
                if not valid_group:
                    break

            if valid_group:
                final_groups.append(group)

    return final_groups