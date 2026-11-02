const DATABASE_NAME = "pairup-footwear";
const DATABASE_VERSION = 1;
const STORE_NAME = "products";

let databasePromise;

export async function saveProduct(product) {
  const database = await openDatabase();
  await runTransaction(database, "readwrite", (store) => store.put(product));
}

export async function loadProducts() {
  const database = await openDatabase();
  return runTransaction(database, "readonly", (store) => store.getAll());
}

export async function deleteProduct(id) {
  const database = await openDatabase();
  await runTransaction(database, "readwrite", (store) => store.delete(id));
}

export async function clearProducts() {
  const database = await openDatabase();
  await runTransaction(database, "readwrite", (store) => store.clear());
}

function openDatabase() {
  if (!databasePromise) {
    databasePromise = new Promise((resolve, reject) => {
      if (!("indexedDB" in window)) {
        reject(new Error("This browser does not support local catalog storage."));
        return;
      }

      const request = indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
      request.onupgradeneeded = () => {
        const database = request.result;
        if (!database.objectStoreNames.contains(STORE_NAME)) {
          database.createObjectStore(STORE_NAME, { keyPath: "id" });
        }
      };
      request.onsuccess = () => {
        const database = request.result;
        database.onversionchange = () => database.close();
        resolve(database);
      };
      request.onerror = () => reject(request.error ?? new Error("Could not open local catalog storage."));
      request.onblocked = () => reject(new Error("Local catalog storage is blocked by another open PairUp tab."));
    }).catch((error) => {
      databasePromise = undefined;
      throw error;
    });
  }
  return databasePromise;
}

function runTransaction(database, mode, operation) {
  return new Promise((resolve, reject) => {
    const transaction = database.transaction(STORE_NAME, mode);
    const request = operation(transaction.objectStore(STORE_NAME));
    let result;
    if (request) {
      request.onsuccess = () => {
        result = request.result;
      };
      request.onerror = () => reject(request.error ?? new Error("Could not access local catalog storage."));
    }
    transaction.oncomplete = () => resolve(result);
    transaction.onerror = () => reject(transaction.error ?? new Error("Could not save the local catalog."));
    transaction.onabort = () => reject(transaction.error ?? new Error("The local catalog operation was cancelled."));
  });
}
