from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
import json
from datetime import datetime, timedelta
from typing import Any
import os
import logging
import psycopg2
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Static reference date for demo consistency (April 21, 2024).
# All date calculations (e.g. days since delivery) in check_return_eligibility
# are relative to this date so the demo produces consistent results regardless
# of when it is actually run.
DEMO_TODAY = datetime(2024, 4, 21)

# Return policy, mirrored from data/return-policy.txt so eligibility, windows,
# fees, and refunds are computed deterministically here instead of by the model.
# Return window (days after delivery) by product category; anything not listed
# uses the standard window.
RETURN_WINDOW_DAYS = {"Electronics": 15}
STANDARD_RETURN_WINDOW_DAYS = 30
# Restocking fee charged on an OPENED return, by category (unopened = 0%).
OPENED_RESTOCKING_FEE = {"Electronics": 0.15}
STANDARD_OPENED_RESTOCKING_FEE = 0.10

# Database connection parameters from environment.
# Use host.containers.internal to reach a host-side PostgreSQL from a container,
# or localhost for a plain Python process on the same machine.
DB_HOST = os.getenv('DB_HOST', 'host.containers.internal')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'techmart')
DB_USER = os.getenv('DB_USER', 'techmart')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'techmart123')
PORT = int(os.getenv("PORT", "9001"))

# Initialize FastMCP server
mcp = FastMCP(
    "TechMart Orders Server",
    host="0.0.0.0",
    port=PORT,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
        allowed_hosts=["*"],
        allowed_origins=["*"],
    )
)

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_db_connection():
    """Open and return a new PostgreSQL connection using RealDictCursor."""
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        cursor_factory=RealDictCursor
    )


def _row_to_order(row: dict) -> dict[str, Any]:
    """Normalise a raw DB row dict to a plain serialisable order dict."""
    order = dict(row)
    if order.get('order_date'):
        order['order_date'] = order['order_date'].strftime('%Y-%m-%d')
    if order.get('delivery_date'):
        order['delivery_date'] = order['delivery_date'].strftime('%Y-%m-%d')
    if 'price' in order:
        order['price'] = str(order['price'])
    return order


def _fetch_all_orders() -> list[dict[str, Any]]:
    """Query all orders from PostgreSQL. Called on every request that needs them."""
    orders = []
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM orders ORDER BY order_date DESC")
                orders = [_row_to_order(dict(row)) for row in cursor.fetchall()]
        logger.info(f"Fetched {len(orders)} orders from database")
    except Exception as e:
        logger.error(f"Error fetching orders from database: {e}")
    return orders


def _fetch_order_by_id(order_id: str) -> dict[str, Any] | None:
    """Query a single order by order_id. Returns None if not found."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM orders WHERE order_id = %s",
                    (order_id,)
                )
                row = cursor.fetchone()
                return _row_to_order(dict(row)) if row else None
    except Exception as e:
        logger.error(f"Error fetching order {order_id} from database: {e}")
        return None


def _normalize_is_opened(value: Any) -> bool:
    """Normalize the is_opened DB value to a Python bool."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == 'yes'


def _normalize_status(value: str) -> tuple[str, str]:
    """
    Return (status_enum, status_display) from a raw DB status string.
    status_enum is uppercase with underscores for machine reasoning.
    status_display is the original human-readable form.
    """
    display = value or ""
    return display.upper().replace(" ", "_"), display


def _order_not_found(order_id: str) -> dict[str, Any]:
    """Return a structured error payload for a missing order."""
    return {
        "success": False,
        "error": "ORDER_NOT_FOUND",
        "message": f"Order {order_id} was not found. Please verify the order ID and try again.",
    }


# ---------------------------------------------------------------------------
# MCP resource
# ---------------------------------------------------------------------------

@mcp.resource("orders://info")
def get_orders_resource() -> str:
    """Expose a live summary of orders as an MCP resource."""
    orders = _fetch_all_orders()
    return json.dumps({
        "orders_count": len(orders),
        "data_source": "PostgreSQL Database",
        "demo_date": DEMO_TODAY.strftime('%Y-%m-%d'),
        "sample_orders": [order['order_id'] for order in orders[:5]],
    })


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------

@mcp.tool()
def get_order(order_id: str) -> dict[str, Any]:
    """
    Look up a single order by its order ID and return its raw details.

    This is a data-only tool. It returns factual order information from the
    database.

    USE THIS TOOL when the customer asks about:
    - Order status or tracking: "Where is my order?", "What is the status of ORD-2024-001?"
    - Delivery information: "When was it delivered?", "Has it arrived?"
    - Product details: "What product did I buy?", "What did I order?"
    - Price or receipt: "How much did I pay?", "What was the cost?"

    For return eligibility ("Can I return ORD-2024-001?"), use the
    check_return_eligibility tool instead — it applies the return policy and does
    all the date and refund math deterministically.

    Args:
        order_id: The order ID to look up, e.g. "ORD-2024-001".

    Returns on success:
        success (True), order_id, customer_email, product_name, category,
        order_date, delivery_date, price (float), status (enum), status_display,
        is_opened (bool).

    Returns on failure:
        success (False), error ("ORDER_NOT_FOUND"), message (str).
    """
    order = _fetch_order_by_id(order_id)
    if order is None:
        return _order_not_found(order_id)

    status_enum, status_display = _normalize_status(order['status'])
    return {
        "success": True,
        "order_id": order['order_id'],
        "customer_email": order['customer_email'],
        "product_name": order['product_name'],
        "category": order['category'],
        "order_date": order['order_date'],
        "delivery_date": order['delivery_date'],
        "price": float(order['price']),
        "status": status_enum,
        "status_display": status_display,
        "is_opened": _normalize_is_opened(order['is_opened']),
    }


@mcp.tool()
def check_return_eligibility(order_id: str) -> dict[str, Any]:
    """
    Determine whether an order can be returned and compute the exact refund.

    This tool applies the TechMart return policy deterministically: it looks up
    the order, computes the return deadline, whether the window is still open as
    of the current demo date, the restocking fee, and the final refund. Prefer
    this tool for ANY "can I return / am I eligible / what's my refund" question
    so the numbers are exact — do not compute dates or refunds yourself.

    Args:
        order_id: The order ID to check, e.g. "ORD-2024-001".

    Returns on success (success=True):
        order_id, product_name, category, price, is_opened, delivery_date,
        today (demo reference date),
        eligible (bool),
        reason (str, present when not eligible),
        return_window_days (int),
        deadline (YYYY-MM-DD, last day a return is accepted),
        days_remaining (int, >=0 when eligible; days left including nothing past
            the deadline),
        days_since_deadline (int, present when the window has closed),
        restocking_fee_percent (int, 0 when unopened),
        restocking_fee_amount (float),
        refund_amount (float).

    Returns on failure:
        success (False), error ("ORDER_NOT_FOUND"), message (str).
    """
    order = _fetch_order_by_id(order_id)
    if order is None:
        return _order_not_found(order_id)

    category = order['category']
    price = float(order['price'])
    is_opened = _normalize_is_opened(order['is_opened'])
    status_enum, status_display = _normalize_status(order['status'])
    delivery_date_str = order['delivery_date']

    result: dict[str, Any] = {
        "success": True,
        "order_id": order['order_id'],
        "product_name": order['product_name'],
        "category": category,
        "price": price,
        "is_opened": is_opened,
        "status": status_display,
        "delivery_date": delivery_date_str,
        "today": DEMO_TODAY.strftime('%Y-%m-%d'),
    }

    # The return window opens on delivery, so gate on the status and not just
    # the delivery date. Orders still in transit or processing carry an
    # *estimated* delivery_date, which would otherwise be mistaken for the day
    # the customer received the item.
    if status_enum != "DELIVERED":
        result.update(
            eligible=False,
            reason=f"Order has not been delivered yet (status: {status_display}).",
        )
        return result

    if not delivery_date_str:
        result.update(
            eligible=False,
            reason="Order is marked delivered but has no delivery date on record.",
        )
        return result

    delivery_date = datetime.strptime(delivery_date_str, '%Y-%m-%d')

    # Restocking fee applies only to opened items and depends on category.
    if is_opened:
        fee_rate = OPENED_RESTOCKING_FEE.get(category, STANDARD_OPENED_RESTOCKING_FEE)
    else:
        fee_rate = 0.0
    restocking_fee_amount = round(price * fee_rate, 2)
    refund_amount = round(price - restocking_fee_amount, 2)

    # Return window and deadline by category.
    window_days = RETURN_WINDOW_DAYS.get(category, STANDARD_RETURN_WINDOW_DAYS)
    deadline = delivery_date + timedelta(days=window_days)
    days_remaining = (deadline - DEMO_TODAY).days

    result.update(
        return_window_days=window_days,
        deadline=deadline.strftime('%Y-%m-%d'),
        restocking_fee_percent=int(round(fee_rate * 100)),
        restocking_fee_amount=restocking_fee_amount,
        refund_amount=refund_amount,
    )

    if days_remaining >= 0:
        result.update(eligible=True, days_remaining=days_remaining)
    else:
        result.update(
            eligible=False,
            days_since_deadline=-days_remaining,
            reason=(
                f"The {window_days}-day return window closed on "
                f"{deadline.strftime('%Y-%m-%d')} ({-days_remaining} days ago)."
            ),
        )

    return result


if __name__ == "__main__":
    logger.info("🚀 Orders MCP Server starting...")
    logger.info(f"🌐 Server will run on http://0.0.0.0:{PORT}")
    logger.info("🔧 Available tools: get_order, check_return_eligibility")
    mcp.run(transport="sse")
