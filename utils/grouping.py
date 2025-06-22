def grouping(minimum_correlation,correlation_matrix,min_group_size=2):
    groups = []

    # For each stock, see if it fits into an existing group
    for stock in correlation_matrix.columns:
        placed = False

        # Check if this stock belongs to any existing group
        for group in groups:
            # If this stock correlates well with ANY stock already in the group
            # could implement a better clustering algorithim
            if any(correlation_matrix.loc[stock, existing_stock] >= minimum_correlation
                   for existing_stock in group if existing_stock != stock):
                group.append(stock)
                placed = True
                break

        # If it doesn't fit anywhere, start a new group
        if not placed:
            groups.append([stock])

    # Clean up - only keep groups with more than 1 stock
    final_groups = [group for group in groups if len(group) > min_group_size]

    return final_groups