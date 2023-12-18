import asyncio
import json
import dash
from dash import html, dcc
from dash.dependencies import Input, Output

# Initialize the Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Received Dictionary Data"),
    html.Div(id='live-update-text'),
    dcc.Interval(
        id='interval-component',
        interval=5*1000,  # in milliseconds (update every 5 seconds)
        n_intervals=0
    )
])

received_data = {}  # Global variable to store received dictionary data

# Update the displayed data based on the received data
@app.callback(
    Output('live-update-text', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_displayed_data(n):
    return json.dumps(received_data, indent=4)

async def handle_client(reader, writer):
    global received_data

    received_data = {}  # Reset received data when a new connection is established

    while True:
        data = await reader.read(4096)
        if not data:
            break
        received_str = data.decode()
        received_data = json.loads(received_str)
        print("Received dictionary from client:")
        print(received_data)

# Run the server
async def run_server():
    server = await asyncio.start_server(
        handle_client, '127.0.0.1', 8888)  # Adjust the IP and port as needed

    async with server:
        await server.serve_forever()

# Run the server and Dash app concurrently
async def main():
    server_task = asyncio.create_task(run_server())
    dash_task = asyncio.create_task(app.run_server(host='127.0.0.1', port=8050, debug=False))
    await asyncio.gather(server_task, dash_task)

if __name__ == '__main__':
    asyncio.run(main())
