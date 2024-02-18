import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series

class DynamicSchema(pa.DataFrameModel):
    # Using a loop to dynamically create column validators
    for i in range(1, 347):  # 347 because range end is exclusive
        locals()[f"N{i}"] = Series[float]

# Example usage
valid_df = pd.DataFrame({f"N{i}": [1.0, 2.0, 3.0] for i in range(1, 347)})

# Validate the DataFrame
validated_df = DynamicSchema.validate(valid_df)
