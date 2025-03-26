from flask import Flask, jsonify, redirect
from flask import jsonify
from flask_cors import CORS
from flask_restful import Api, MethodNotAllowed, NotFound
from flask_swagger_ui import get_swaggerui_blueprint
from util.common import domain, port, prefix, build_swagger_config_json
from resources.swaggerConfig import SwaggerConfig
from resources.modelResource import modelsPOSTResource

# ============================================
# Main
# ============================================
application = Flask(__name__)
app = application
app.config['PROPAGATE_EXCEPTIONS'] = True
CORS(app, resources={r"/*": {"origins": "*"}})
api = Api(app, prefix=prefix, catch_all_404s=True)

# ============================================
# Swagger
# ============================================
build_swagger_config_json()
swaggerui_blueprint = get_swaggerui_blueprint(
    prefix,
    f'https://{domain}{prefix}/swagger-config',
    config={
        'app_name': "Flask API",
        "layout": "BaseLayout",
        "docExpansion": "none"
    },
)
app.register_blueprint(swaggerui_blueprint)

# ============================================
# Error Handler
# ============================================


@app.errorhandler(NotFound)
def handle_method_not_found(e):
    response = jsonify({"message": str(e)})
    response.status_code = 404
    return response

@app.errorhandler(MethodNotAllowed)
def handle_method_not_allowed_error(e):
    response = jsonify({"message": str(e)})
    response.status_code = 405
    return response

@app.route('/')
def redirect_to_prefix():
    if prefix != '':
        return redirect(prefix)

# ============================================
# Add Resource
# ============================================
# GET swagger config
api.add_resource(SwaggerConfig, '/swagger-config')
# GET books
api.add_resource(modelsPOSTResource, '/predict')

if __name__ == '__main__':
    app.run(debug=True)