from datasets import Dataset

from rules_lawyer_models.utils import logging_protocol

# import textwrap


# def dump_first_row(dataset, wrap_width=80):
#     """Pretty-print the first row of a HuggingFace Dataset."""
#     row = dataset[0]
#     for key, value in row.items():
#         print(f"--- {key} ---")
#         if isinstance(value, str) and len(value) > wrap_width:
#             print(textwrap.fill(value, width=wrap_width))
#         else:
#             print(value)
#         print()


def dump_first_row_to_logger(dataset: Dataset, logger: logging_protocol.LoggingProtocol) -> None:
    row = dataset[0]
    logger.report_table_message(row)
