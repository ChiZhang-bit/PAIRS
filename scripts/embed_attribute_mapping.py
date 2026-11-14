#!/usr/bin/env python3
"""Convert URFATTR v2/v3 to self-contained v4 without changing stored columns."""
import argparse
import json
import os
from pathlib import Path
import stat
import struct
import tempfile


HEADER = struct.Struct('<8sIIQII')
DESCRIPTOR = struct.Struct('<16sBBHI')


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate mapping key: {key}')
        result[key] = value
    return result


def validate_mapping(raw, categories):
    text = raw.decode('utf-8')
    # Match the native dictionary parser's escape vocabulary. The original
    # JSON bytes are embedded unchanged, including whitespace and line endings.
    inside, escaped = False, False
    for char in text:
        if escaped:
            if char not in '"\\/bfnrt':
                raise ValueError('Mapping uses an escape unsupported by the native reader')
            escaped = False
        elif inside and char == '\\':
            escaped = True
        elif char == '"':
            inside = not inside
    mapping = json.loads(text, object_pairs_hook=unique_object)
    if not isinstance(mapping, dict) or set(mapping) != set(categories):
        raise ValueError('Mapping fields do not match categorical descriptors')
    for field, cardinality in categories.items():
        dictionary = mapping[field]
        if not isinstance(dictionary, dict) or set(dictionary) != {str(i) for i in range(cardinality)}:
            raise ValueError(f'Mapping codes are not contiguous or have the wrong count: {field}')
        labels = list(dictionary.values())
        if any(not isinstance(label, str) or not label for label in labels) or len(set(labels)) != len(labels):
            raise ValueError(f'Mapping labels must be nonempty and unique: {field}')


def inspect_attributes(stream, size):
    header = stream.read(HEADER.size)
    if len(header) != HEADER.size:
        raise ValueError('Truncated attribute header')
    magic, version, header_bytes, rows, numeric, categorical = HEADER.unpack(header)
    if magic != b'URFATTR\0' or version not in (2, 3):
        raise ValueError('Conversion requires a URFATTR v2 or v3 attribute file')
    if not 0 < rows <= 0xFFFFFFFF or not numeric or not categorical or numeric + categorical > 4096:
        raise ValueError('Invalid attribute shape')
    if header_bytes != HEADER.size + DESCRIPTOR.size * (numeric + categorical):
        raise ValueError('Invalid attribute header size')
    names, categories, row_bytes = set(), {}, 0
    for field in range(numeric + categorical):
        descriptor = stream.read(DESCRIPTOR.size)
        if len(descriptor) != DESCRIPTOR.size:
            raise ValueError('Truncated attribute descriptor')
        raw_name, kind, dtype, reserved, cardinality = DESCRIPTOR.unpack(descriptor)
        end = raw_name.find(b'\0')
        if end <= 0 or any(raw_name[end:]):
            raise ValueError('Invalid attribute name padding')
        name = raw_name[:end].decode('ascii')
        if name in names or reserved or kind != (1 if field < numeric else 2):
            raise ValueError('Invalid or duplicate attribute descriptor')
        names.add(name)
        if kind == 1:
            if dtype not in (1, 2) or cardinality:
                raise ValueError('Invalid numerical descriptor')
            row_bytes += 4
        else:
            if dtype not in (3, 4) or (version == 2 and dtype == 4):
                raise ValueError('Invalid categorical storage type')
            if not 0 < cardinality <= (256 if dtype == 3 else 65536):
                raise ValueError('Invalid dictionary size')
            categories[name] = cardinality
            row_bytes += 1 if dtype == 3 else 2
    payload_end = header_bytes + rows * row_bytes
    if size != payload_end:
        raise ValueError('Attribute payload length mismatch')
    return header, version, rows, categories, payload_end


def convert(attributes, mapping, output=None):
    source, mapping = Path(attributes).resolve(), Path(mapping).resolve()
    destination = source if output is None else Path(output).resolve()
    if destination != source and destination.exists():
        raise FileExistsError(f'Output already exists: {destination}')
    raw_mapping = mapping.read_bytes()
    if not 0 < len(raw_mapping) <= 0xFFFFFFFF:
        raise ValueError('Embedded mapping length must fit a nonzero uint32')
    before = source.stat()
    temporary = None
    try:
        with source.open('rb') as incoming:
            header, version, rows, categories, payload_end = inspect_attributes(incoming, before.st_size)
            validate_mapping(raw_mapping, categories)
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, name = tempfile.mkstemp(prefix=destination.name + '.', suffix='.tmp', dir=destination.parent)
            temporary = Path(name)
            with os.fdopen(descriptor, 'wb') as outgoing:
                updated = bytearray(header)
                struct.pack_into('<I', updated, 8, 4)
                outgoing.write(updated)
                incoming.seek(HEADER.size)
                remaining = payload_end - HEADER.size
                while remaining:
                    chunk = incoming.read(min(remaining, 1024 * 1024))
                    if not chunk:
                        raise ValueError('Attribute source was truncated during conversion')
                    outgoing.write(chunk)
                    remaining -= len(chunk)
                outgoing.write(struct.pack('<I', len(raw_mapping)))
                outgoing.write(raw_mapping)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            after = source.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                raise ValueError('Attribute source changed during conversion')
        if temporary.stat().st_size != payload_end + 4 + len(raw_mapping):
            raise ValueError('Converted attribute length mismatch')
        os.chmod(temporary, stat.S_IMODE(before.st_mode))
        # The temporary file lives beside the destination, so replacement does
        # not expose a partially written attribute file to the next reader.
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {'source_version': version, 'version': 4, 'rows': rows,
            'payload_end': payload_end, 'embedded_mapping_bytes': len(raw_mapping),
            'output': str(destination), 'mapping_retained': str(mapping)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attributes', type=Path, required=True)
    parser.add_argument('--mapping', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='New output file; omitted means atomic in-place conversion')
    args = parser.parse_args()
    try:
        result = convert(args.attributes, args.mapping, args.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f'{error}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
