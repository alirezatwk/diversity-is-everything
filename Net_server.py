#!/usr/bin/python

import argparse
import requests
from getpass import getpass


def run(username, password):
    url = 'https://internet.ut.ac.ir/login?dst=http%3A%2F%2Fgoogle.com%2F'
    data = {'username': username, 'password': password}
    response = requests.post(url, data=data)
    print(response.text)

def main():
    parser = argparse.ArgumentParser('Login into UT internet.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-u', '--username', type=str, help='Username.')
    # parser.add_argument('-p', '--password', type=str, help='Password.')

    args = parser.parse_args()
    password = getpass()
    run(username=args.username, password=password)

if __name__ == '__main__':
    main()