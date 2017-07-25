


def transform(org, destination):
    outsize = 0
    with open(org, 'rb') as infile:
        content = infile.read()

    with open(destination, 'wb') as output:
        for line in content.splitlines():
            outsize += len(line) + 1
            output.write(line + str.encode('\n'))
