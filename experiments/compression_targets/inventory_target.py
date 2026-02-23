from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Tuple
import json

class TransactionType(Enum):
    """Types of inventory transactions for audit trail."""
    ADD = "add"
    REMOVE = "remove"
    UPDATE = "update"
    TRANSFER_OUT = "transfer_out"
    TRANSFER_IN = "transfer_in"

class Product:
    """Represents a stockable product with pricing and quantity tracking."""
    def __init__(self, sku: str, name: str, price: float, quantity: int,
                 category: str = "general", reorder_point: int = 5):
        self.sku = sku
        self.name = name
        self.price = price
        self.quantity = quantity
        self.category = category
        self.reorder_point = reorder_point
        self.created_at = datetime.now()
        self.last_modified = datetime.now()

    @property
    def total_value(self) -> float:
        """Total value of this product's stock."""
        return self.price * self.quantity

    @property
    def needs_reorder(self) -> bool:
        """Check if quantity has fallen below reorder point."""
        return self.quantity <= self.reorder_point

    def to_dict(self) -> dict:
        """Serialize product to dictionary for JSON export."""
        return {
            "sku": self.sku, "name": self.name,
            "price": self.price, "quantity": self.quantity,
            "category": self.category, "value": self.total_value,
        }

class Warehouse:
    """
    Manages inventory for a single warehouse location.
    Tracks all mutations through an immutable transaction log.
    """
    def __init__(self, name: str, location: str = "default"):
        self.name = name
        self.location = location
        self.inventory: Dict[str, Product] = {}
        self._transaction_log: List[Tuple[TransactionType, str, int, datetime]] = []

    def _log_transaction(self, tx_type: TransactionType, sku: str, qty: int) -> None:
        """Append an immutable transaction record."""
        self._transaction_log.append((tx_type, sku, qty, datetime.now()))

    def add_product(self, product: Product) -> bool:
        """
        Add a new product to inventory. Returns False if SKU already exists.
        """
        if product.sku in self.inventory:
            return False
        self.inventory[product.sku] = product
        self._log_transaction(TransactionType.ADD, product.sku, product.quantity)
        return True

    def remove_product(self, sku: str) -> Optional[Product]:
        """Remove a product entirely. Returns the removed product or None."""
        product = self.inventory.pop(sku, None)
        if product:
            self._log_transaction(TransactionType.REMOVE, sku, 0)
        return product

    def update_quantity(self, sku: str, delta: int) -> bool:
        """
        Adjust quantity by delta. Prevents negative stock.
        Returns True if the update was applied.
        """
        if sku not in self.inventory:
            return False
        product = self.inventory[sku]
        new_qty = product.quantity + delta
        if new_qty < 0:
            return False  # Reject rather than clamp to zero
        product.quantity = new_qty
        product.last_modified = datetime.now()
        self._log_transaction(TransactionType.UPDATE, sku, delta)
        return True

    def get_total_value(self) -> float:
        """Calculate total inventory value across all products."""
        return sum(p.total_value for p in self.inventory.values())

    def get_low_stock(self, threshold: Optional[int] = None) -> List[Product]:
        """
        Return products below their reorder point (or a custom threshold).
        """
        return [
            p for p in self.inventory.values()
            if p.quantity <= (threshold if threshold is not None else p.reorder_point)
        ]

    def get_products_by_category(self, category: str) -> List[Product]:
        """Filter inventory by product category."""
        return [p for p in self.inventory.values() if p.category == category]

    def transfer_to(self, destination: 'Warehouse', sku: str, amount: int) -> bool:
        """
        Transfer stock from this warehouse to another.
        Atomic: both sides are updated or neither is.
        """
        if sku not in self.inventory or self.inventory[sku].quantity < amount:
            return False
        # Ensure destination has the product entry
        if sku not in destination.inventory:
            source_product = self.inventory[sku]
            destination.add_product(Product(
                sku, source_product.name, source_product.price, 0,
                source_product.category, source_product.reorder_point,
            ))
        self.inventory[sku].quantity -= amount
        self._log_transaction(TransactionType.TRANSFER_OUT, sku, amount)
        destination.inventory[sku].quantity += amount
        destination._log_transaction(TransactionType.TRANSFER_IN, sku, amount)
        return True

class InventoryReport:
    """Generates formatted reports across multiple warehouse locations."""
    def __init__(self, warehouses: List[Warehouse]):
        self.warehouses = warehouses
        self.generated_at = datetime.now()

    def total_portfolio_value(self) -> float:
        """Total value across all warehouses."""
        return sum(w.get_total_value() for w in self.warehouses)

    def generate_text(self) -> str:
        """Generate a human-readable inventory report."""
        lines = [f"=== Inventory Report ({self.generated_at:%Y-%m-%d %H:%M}) ==="]
        for w in self.warehouses:
            lines.append(f"\nWarehouse: {w.name} ({w.location})")
            lines.append(f"  Products: {len(w.inventory)}")
            lines.append(f"  Value: ${w.get_total_value():,.2f}")
            for p in sorted(w.inventory.values(), key=lambda x: x.sku):
                status = " ⚠️" if p.needs_reorder else ""
                lines.append(
                    f"    {p.sku}: {p.name} x{p.quantity} "
                    f"@ ${p.price:.2f} = ${p.total_value:,.2f}{status}"
                )
            alerts = w.get_low_stock()
            if alerts:
                lines.append("  REORDER ALERTS:")
                for p in alerts:
                    lines.append(
                        f"    {p.sku}: {p.name} "
                        f"({p.quantity} remaining, reorder at {p.reorder_point})"
                    )
        lines.append(f"\nTotal Portfolio Value: ${self.total_portfolio_value():,.2f}")
        return "\n".join(lines)

    def export_json(self) -> str:
        """Export inventory data as JSON."""
        data = {
            "generated_at": self.generated_at.isoformat(),
            "warehouses": {
                w.name: {
                    "location": w.location,
                    "products": [p.to_dict() for p in w.inventory.values()],
                    "total_value": w.get_total_value(),
                }
                for w in self.warehouses
            },
            "portfolio_value": self.total_portfolio_value(),
        }
        return json.dumps(data, indent=2)

if __name__ == "__main__":
    w1 = Warehouse("Main", location="Building A")
    w2 = Warehouse("Overflow", location="Building B")
    w1.add_product(Product("SKU001", "Widget A", 12.99, 100, "widgets"))
    w1.add_product(Product("SKU002", "Widget B", 24.50, 3, "widgets", reorder_point=10))
    w1.add_product(Product("SKU003", "Gadget C", 99.99, 50, "gadgets"))
    w1.transfer_to(w2, "SKU001", 30)
    w1.update_quantity("SKU002", -1)
    report = InventoryReport([w1, w2])
    print(report.generate_text())
    print("\n--- JSON Export ---")
    print(report.export_json())
