from datetime import datetime

class Product:
    def __init__(self, sku, name, price, quantity):
        self.sku = sku
        self.name = name
        self.price = price
        self.quantity = quantity
        self.created = datetime.now()

class Warehouse:
    def __init__(self, name):
        self.name = name
        self.inventory = {}
        self.transactions = []

    def add_product(self, product):
        self.inventory[product.sku] = product
        self.transactions.append(("ADD", product.sku, product.quantity, datetime.now()))

    def remove_product(self, sku):
        if sku in self.inventory:
            del self.inventory[sku]
            self.transactions.append(("REMOVE", sku, 0, datetime.now()))

    def update_quantity(self, sku, delta):
        if sku in self.inventory:
            self.inventory[sku].quantity += delta
            if self.inventory[sku].quantity < 0:
                self.inventory[sku].quantity = 0
            self.transactions.append(("UPDATE", sku, delta, datetime.now()))

    def get_value(self):
        total = 0.0
        for p in self.inventory.values():
            total += p.price * p.quantity
        return total

    def low_stock(self, threshold=5):
        result = []
        for p in self.inventory.values():
            if p.quantity <= threshold:
                result.append(p)
        return result

    def transfer(self, other, sku, amount):
        if sku in self.inventory and self.inventory[sku].quantity >= amount:
            self.update_quantity(sku, -amount)
            if sku not in other.inventory:
                p = self.inventory[sku]
                other.add_product(Product(sku, p.name, p.price, 0))
            other.update_quantity(sku, amount)
            return True
        return False

class InventoryReport:
    def __init__(self, warehouses):
        self.warehouses = warehouses

    def total_value(self):
        return sum(w.get_value() for w in self.warehouses)

    def generate(self):
        print("=== Inventory Report ===")
        for w in self.warehouses:
            print(f"\nWarehouse: {w.name}")
            print(f"  Products: {len(w.inventory)}")
            print(f"  Value: ${w.get_value():.2f}")
            for p in w.inventory.values():
                print(f"    {p.sku}: {p.name} x{p.quantity} @ ${p.price:.2f}")
            alerts = w.low_stock()
            if alerts:
                print(f"  LOW STOCK:")
                for p in alerts:
                    print(f"    {p.sku}: {p.name} ({p.quantity} remaining)")
        print(f"\nTotal Portfolio Value: ${self.total_value():.2f}")

if __name__ == "__main__":
    w1 = Warehouse("Main")
    w2 = Warehouse("Overflow")
    w1.add_product(Product("SKU001", "Widget A", 12.99, 100))
    w1.add_product(Product("SKU002", "Widget B", 24.50, 3))
    w1.add_product(Product("SKU003", "Gadget C", 99.99, 50))
    w1.transfer(w2, "SKU001", 30)
    w1.update_quantity("SKU002", -1)
    report = InventoryReport([w1, w2])
    report.generate()
